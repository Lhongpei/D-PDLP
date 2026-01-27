#include "cupdlpx.h"
#include "nccl.h"
#include "distributed_op.h"
#include "pdlp_core_op.h"
#include "distribution_utils.h"
#include "internal_types.h"
#include "preconditioner.h"
#include "presolve.h"
#include "solver.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>



void compute_next_pdhg_primal_solution_distributed(pdhg_solver_state_t *state)
{
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol,
                                          state->current_dual_solution));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));
    
    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    NCCL_CHECK(ncclAllReduce(
        (const void *)state->dual_product, 
        (void *)state->dual_product,      
        state->num_variables,              
        ncclDouble,                   
        ncclSum,                         
        state->grid_context->nccl_col,    
        0                                
    ));

    double step = state->step_size / state->primal_weight;

    if (state->is_this_major_iteration ||
        ((state->total_count + 2) %
         get_print_frequency(state->total_count + 2)) == 0)
    {
        compute_next_pdhg_primal_solution_major_kernel<<<state->num_blocks_primal,
                                                         THREADS_PER_BLOCK>>>(
            state->current_primal_solution, state->pdhg_primal_solution,
            state->reflected_primal_solution, state->dual_product,
            state->objective_vector, state->variable_lower_bound,
            state->variable_upper_bound, state->num_variables, step,
            state->dual_slack);
    }
    else
    {
        compute_next_pdhg_primal_solution_kernel<<<state->num_blocks_primal,
                                                   THREADS_PER_BLOCK>>>(
            state->current_primal_solution, state->reflected_primal_solution,
            state->dual_product, state->objective_vector,
            state->variable_lower_bound, state->variable_upper_bound,
            state->num_variables, step);
    }
}

void compute_next_pdhg_dual_solution_distributed(pdhg_solver_state_t *state)
{
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol,
                                          state->reflected_primal_solution));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    NCCL_CHECK(ncclAllReduce(
        (const void *)state->primal_product, 
        (void *)state->primal_product,
        state->num_constraints,
        ncclDouble,
        ncclSum,
        state->grid_context->nccl_row,
        0
    ));

    double step = state->step_size * state->primal_weight;

    if (state->is_this_major_iteration ||
        ((state->total_count + 2) %
         get_print_frequency(state->total_count + 2)) == 0)
    {
        compute_next_pdhg_dual_solution_major_kernel<<<state->num_blocks_dual,
                                                       THREADS_PER_BLOCK>>>(
            state->current_dual_solution, state->pdhg_dual_solution,
            state->reflected_dual_solution, state->primal_product,
            state->constraint_lower_bound, state->constraint_upper_bound,
            state->num_constraints, step);
    }
    else
    {
        compute_next_pdhg_dual_solution_kernel<<<state->num_blocks_dual,
                                                 THREADS_PER_BLOCK>>>(
            state->current_dual_solution, state->reflected_dual_solution,
            state->primal_product, state->constraint_lower_bound,
            state->constraint_upper_bound, state->num_constraints, step);
    }
}

void compute_fixed_point_error_distributed(pdhg_solver_state_t *state)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual,
                                    THREADS_PER_BLOCK>>>(
        state->current_primal_solution, state->reflected_primal_solution,
        state->delta_primal_solution, state->current_dual_solution,
        state->reflected_dual_solution, state->delta_dual_solution,
        state->num_variables, state->num_constraints);

    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_dual_sol, state->delta_dual_solution));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    double local_primal_norm = 0.0;
    double local_dual_norm = 0.0;
    double local_cross_term = 0.0;

    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints,
                                   state->delta_dual_solution, 1, &local_dual_norm));
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables,
                                   state->delta_primal_solution, 1,
                                   &local_primal_norm));
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables,
                            state->dual_product, 1, state->delta_primal_solution,
                            1, &local_cross_term));

    double local_primal_sq = local_primal_norm * local_primal_norm;
    double local_dual_sq = local_dual_norm * local_dual_norm;

    double global_primal_sq = 0.0;
    double global_dual_sq = 0.0;
    double global_cross_term = 0.0;

    MPI_Allreduce(&local_primal_sq, &global_primal_sq, 1, MPI_DOUBLE, MPI_SUM, 
                  state->grid_context->comm_row);

    MPI_Allreduce(&local_dual_sq, &global_dual_sq, 1, MPI_DOUBLE, MPI_SUM, 
                  state->grid_context->comm_col);

    MPI_Allreduce(&local_cross_term, &global_cross_term, 1, MPI_DOUBLE, MPI_SUM, 
                  state->grid_context->comm_global);

    double movement = global_primal_sq * state->primal_weight +
                      global_dual_sq / state->primal_weight;
    
    double interaction = 2 * state->step_size * global_cross_term;

    state->fixed_point_error = sqrt(movement + interaction);
}

void perform_restart_distributed(pdhg_solver_state_t *state,
                            const pdhg_parameters_t *params)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual,
                                    THREADS_PER_BLOCK>>>(
        state->initial_primal_solution, state->pdhg_primal_solution,
        state->delta_primal_solution, state->initial_dual_solution,
        state->pdhg_dual_solution, state->delta_dual_solution,
        state->num_variables, state->num_constraints);

    double primal_dist, dual_dist;
    
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables,
                                   state->delta_primal_solution, 1,
                                   &primal_dist));
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints,
                                   state->delta_dual_solution, 1, &dual_dist));

    double local_primal_sq = primal_dist * primal_dist;
    double local_dual_sq = dual_dist * dual_dist;

    double global_primal_sq = 0.0;
    double global_dual_sq = 0.0;

    MPI_Allreduce(&local_primal_sq, &global_primal_sq, 1, MPI_DOUBLE, MPI_SUM, 
                  state->grid_context->comm_row);

    MPI_Allreduce(&local_dual_sq, &global_dual_sq, 1, MPI_DOUBLE, MPI_SUM, 
                  state->grid_context->comm_col);

    primal_dist = sqrt(global_primal_sq);
    dual_dist = sqrt(global_dual_sq);

    double ratio_infeas =
        state->relative_dual_residual / state->relative_primal_residual;

    if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 &&
        dual_dist < 1e12 && ratio_infeas > 1e-8 && ratio_infeas < 1e8)
    {
        double error =
            log(dual_dist) - log(primal_dist) - log(state->primal_weight);
        state->primal_weight_error_sum *= params->restart_params.i_smooth;
        state->primal_weight_error_sum += error;
        double delta_error = error - state->primal_weight_last_error;
        state->primal_weight *=
            exp(params->restart_params.k_p * error +
                params->restart_params.k_i * state->primal_weight_error_sum +
                params->restart_params.k_d * delta_error);
        state->primal_weight_last_error = error;
    }
    else
    {
        state->primal_weight = state->best_primal_weight;
        state->primal_weight_error_sum = 0.0;
        state->primal_weight_last_error = 0.0;
    }

    double primal_dual_residual_gap = abs(
        log10(state->relative_dual_residual / state->relative_primal_residual));
    if (primal_dual_residual_gap < state->best_primal_dual_residual_gap)
    {
        state->best_primal_dual_residual_gap = primal_dual_residual_gap;
        state->best_primal_weight = state->primal_weight;
    }

    CUDA_CHECK(cudaMemcpy(
        state->initial_primal_solution, state->pdhg_primal_solution,
        state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->current_primal_solution, state->pdhg_primal_solution,
        state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, state->pdhg_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->pdhg_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

void compute_residual_distributed(pdhg_solver_state_t *state, norm_type_t optimality_norm)
{
    cusparseDnVecSetValues(state->vec_primal_sol, state->pdhg_primal_solution);
    cusparseDnVecSetValues(state->vec_dual_sol, state->pdhg_dual_solution);
    cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product);
    cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product);

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    NCCL_CHECK(ncclAllReduce((const void *)state->primal_product,
                             (void *)state->primal_product,
                             state->num_constraints, ncclDouble, ncclSum,
                             state->grid_context->nccl_row, 0));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));
    
    NCCL_CHECK(ncclAllReduce((const void *)state->dual_product,
                             (void *)state->dual_product,
                             state->num_variables, ncclDouble, ncclSum,
                             state->grid_context->nccl_col, 0));

    compute_residual_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
        state->primal_residual, state->primal_product,
        state->constraint_lower_bound, state->constraint_upper_bound,
        state->pdhg_dual_solution, state->dual_residual, state->dual_product,
        state->dual_slack, state->objective_vector, state->constraint_rescaling,
        state->variable_rescaling, state->primal_slack,
        state->constraint_lower_bound_finite_val,
        state->constraint_upper_bound_finite_val, state->num_constraints,
        state->num_variables);

    double local_primal_res = 0.0;
    double global_primal_res = 0.0;

    if (optimality_norm == NORM_TYPE_L_INF) {
        local_primal_res = get_vector_inf_norm(state->blas_handle, 
                                               state->num_constraints, state->primal_residual);
        MPI_Allreduce(&local_primal_res, &global_primal_res, 1, MPI_DOUBLE, MPI_MAX, 
                      state->grid_context->comm_col);
        state->absolute_primal_residual = global_primal_res;
    } else {
        CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, 
                                       state->primal_residual, 1, 
                                       &local_primal_res));
        double local_sq = local_primal_res * local_primal_res;
        MPI_Allreduce(&local_sq, &global_primal_res, 1, MPI_DOUBLE, MPI_SUM, 
                      state->grid_context->comm_col);
        state->absolute_primal_residual = sqrt(global_primal_res);
    }

    state->absolute_primal_residual /= state->constraint_bound_rescaling;

    double local_dual_res = 0.0;
    double global_dual_res = 0.0;

    if (optimality_norm == NORM_TYPE_L_INF) {
        local_dual_res = get_vector_inf_norm(state->blas_handle, 
                                             state->num_variables, state->dual_residual);
        MPI_Allreduce(&local_dual_res, &global_dual_res, 1, MPI_DOUBLE, MPI_MAX, 
                      state->grid_context->comm_row);
        state->absolute_dual_residual = global_dual_res;
    } else {
        CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables,
                                       state->dual_residual, 1,
                                       &local_dual_res));
        double local_sq = local_dual_res * local_dual_res;
        MPI_Allreduce(&local_sq, &global_dual_res, 1, MPI_DOUBLE, MPI_SUM, 
                      state->grid_context->comm_row);
        state->absolute_dual_residual = sqrt(global_dual_res);
    }

    state->absolute_dual_residual /= state->objective_vector_rescaling;

    double local_primal_obj;
    CUBLAS_CHECK(cublasDdot(
        state->blas_handle, state->num_variables, state->objective_vector, 1,
        state->pdhg_primal_solution, 1, &local_primal_obj));
    
    double global_primal_obj;
    MPI_Allreduce(&local_primal_obj, &global_primal_obj, 1, MPI_DOUBLE, MPI_SUM, 
                  state->grid_context->comm_row);

    state->primal_objective_value =
        global_primal_obj / (state->constraint_bound_rescaling *
                             state->objective_vector_rescaling) +
        state->objective_constant;

    double local_base_dual;
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables,
                            state->dual_slack, 1, state->pdhg_primal_solution, 1,
                            &local_base_dual));
    double global_base_dual;
    MPI_Allreduce(&local_base_dual, &global_base_dual, 1, MPI_DOUBLE, MPI_SUM, 
                  state->grid_context->comm_row);

    double local_dual_slack_sum =
        get_vector_sum(state->blas_handle, state->num_constraints,
                       state->ones_dual_d, state->primal_slack);
    
    double global_dual_slack_sum;
    MPI_Allreduce(&local_dual_slack_sum, &global_dual_slack_sum, 1, MPI_DOUBLE, MPI_SUM, 
                  state->grid_context->comm_col);

    state->dual_objective_value = (global_base_dual + global_dual_slack_sum) /
                                      (state->constraint_bound_rescaling *
                                       state->objective_vector_rescaling) +
                                  state->objective_constant;

    state->relative_primal_residual = 
        state->absolute_primal_residual / (1.0 + state->constraint_bound_norm);
    
    state->relative_dual_residual =
        state->absolute_dual_residual / (1.0 + state->objective_vector_norm);

    state->objective_gap =
        fabs(state->primal_objective_value - state->dual_objective_value);

    state->relative_objective_gap =
        state->objective_gap / (1.0 + fabs(state->primal_objective_value) +
                                fabs(state->dual_objective_value));
}


void compute_infeasibility_information_distributed(pdhg_solver_state_t *state)
{
    primal_infeasibility_project_kernel<<<state->num_blocks_primal,
                                          THREADS_PER_BLOCK>>>(
        state->delta_primal_solution, state->variable_lower_bound,
        state->variable_upper_bound, state->num_variables);
    dual_infeasibility_project_kernel<<<state->num_blocks_dual,
                                        THREADS_PER_BLOCK>>>(
        state->delta_dual_solution, state->constraint_lower_bound,
        state->constraint_upper_bound, state->num_constraints);

    double local_primal_ray_inf_norm = get_vector_inf_norm(
        state->blas_handle, state->num_variables, state->delta_primal_solution);
    double global_primal_ray_inf_norm = 0.0;
    MPI_Allreduce(&local_primal_ray_inf_norm, &global_primal_ray_inf_norm, 1, MPI_DOUBLE, MPI_MAX, state->grid_context->comm_row);

    if (global_primal_ray_inf_norm > 0.0)
    {
        double scale = 1.0 / global_primal_ray_inf_norm;
        cublasDscal(state->blas_handle, state->num_variables, &scale,
                    state->delta_primal_solution, 1);
    }

    double local_dual_ray_inf_norm = get_vector_inf_norm(
        state->blas_handle, state->num_constraints, state->delta_dual_solution);
    double dual_ray_inf_norm = 0.0;
    MPI_Allreduce(&local_dual_ray_inf_norm, &dual_ray_inf_norm, 1, MPI_DOUBLE, MPI_MAX, state->grid_context->comm_col);

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol,
                                          state->delta_primal_solution));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_dual_sol, state->delta_dual_solution));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    NCCL_CHECK(ncclAllReduce((const void *)state->primal_product,
                             (void *)state->primal_product,
                             state->num_constraints, ncclDouble, ncclSum,
                             state->grid_context->nccl_row, 0));
    
    NCCL_CHECK(ncclAllReduce((const void *)state->dual_product,
                             (void *)state->dual_product,
                             state->num_variables, ncclDouble, ncclSum,
                             state->grid_context->nccl_col, 0));

    double local_primal_ray_linear_objective = 0.0;
    CUBLAS_CHECK(cublasDdot(
        state->blas_handle, state->num_variables, state->objective_vector, 1,
        state->delta_primal_solution, 1, &local_primal_ray_linear_objective));
    local_primal_ray_linear_objective /=
        (state->constraint_bound_rescaling * state->objective_vector_rescaling);
    MPI_Allreduce(&local_primal_ray_linear_objective, &state->primal_ray_linear_objective, 1, MPI_DOUBLE, MPI_SUM, state->grid_context->comm_row);

    dual_solution_dual_objective_contribution_kernel<<<state->num_blocks_dual,
                                                       THREADS_PER_BLOCK>>>(
        state->constraint_lower_bound_finite_val,
        state->constraint_upper_bound_finite_val, state->delta_dual_solution,
        state->num_constraints, state->primal_slack);

    dual_objective_dual_slack_contribution_array_kernel<<<
        state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->dual_product, state->dual_slack,
        state->variable_lower_bound_finite_val,
        state->variable_upper_bound_finite_val, state->num_variables);

    double sum_primal_slack = 0.0;
    double sum_dual_slack = 0.0;
    double local_sum_primal_slack =
        get_vector_sum(state->blas_handle, state->num_constraints,
                       state->ones_dual_d, state->primal_slack);
    double local_sum_dual_slack =
        get_vector_sum(state->blas_handle, state->num_variables,
                       state->ones_primal_d, state->dual_slack);
    MPI_Allreduce(&local_sum_primal_slack, &sum_primal_slack, 1, MPI_DOUBLE, MPI_SUM, state->grid_context->comm_col);
    MPI_Allreduce(&local_sum_dual_slack, &sum_dual_slack, 1, MPI_DOUBLE, MPI_SUM, state->grid_context->comm_row);

    state->dual_ray_objective =
        (sum_primal_slack + sum_dual_slack) /
        (state->constraint_bound_rescaling * state->objective_vector_rescaling);

    compute_primal_infeasibility_kernel<<<state->num_blocks_dual,
                                          THREADS_PER_BLOCK>>>(
        state->primal_product, state->constraint_lower_bound,
        state->constraint_upper_bound, state->num_constraints,
        state->primal_slack, state->constraint_rescaling);
    compute_dual_infeasibility_kernel<<<state->num_blocks_primal,
                                        THREADS_PER_BLOCK>>>(
        state->dual_product, state->variable_lower_bound,
        state->variable_upper_bound, state->num_variables, state->dual_slack,
        state->variable_rescaling);

    double primal_slack_norm = 0.0;
    double dual_slack_norm = 0.0;
    double local_primal_slack_norm = get_vector_inf_norm(
        state->blas_handle, state->num_constraints, state->primal_slack);
    double local_dual_slack_norm = get_vector_inf_norm(
        state->blas_handle, state->num_variables, state->dual_slack);
    MPI_Allreduce(&local_primal_slack_norm, &primal_slack_norm, 1, MPI_DOUBLE, MPI_MAX, state->grid_context->comm_row);
    MPI_Allreduce(&local_dual_slack_norm, &dual_slack_norm, 1, MPI_DOUBLE, MPI_MAX, state->grid_context->comm_col);

    state->max_dual_ray_infeasibility = dual_slack_norm;
    state->max_primal_ray_infeasibility = primal_slack_norm;
    double scaling_factor = fmax(dual_ray_inf_norm, dual_slack_norm);
    if (scaling_factor > 0.0)
    {
        state->max_dual_ray_infeasibility /= scaling_factor;
        state->dual_ray_objective /= scaling_factor;
    }
    else
    {
        state->max_dual_ray_infeasibility = 0.0;
        state->dual_ray_objective = 0.0;
    }
}