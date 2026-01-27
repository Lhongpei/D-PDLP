/*
Copyright 2025 Haihao Lu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "cupdlpx.h"
#include "permute.h"
#include "distribution_utils.h"
#include "distributed_solver.h"
#include "distributed_op.h"
#include "internal_types.h"
#include "preconditioner.h"
#include "presolve.h"
#include "solver.h"
#include "pdlp_core_op.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <float.h>
void get_best_grid_dims(int m, int n, int n_procs, int *out_r, int *out_c);
static void allreduce_obj_bound_norm(pdhg_solver_state_t *state, const pdhg_parameters_t *params);
cupdlpx_result_t *create_result_from_state_distributed(pdhg_solver_state_t *state, const lp_problem_t *original_problem);
static cupdlpx_result_t *distributed_optimize_core(const pdhg_parameters_t *params, const lp_problem_t *original_problem, grid_context_t *grid_context);
static void select_valid_grid_size(const pdhg_parameters_t *params, const lp_problem_t *original_problem, pdhg_parameters_t *sub_params);
static lp_problem_t *permute_lp_problem(const pdhg_parameters_t *params, const lp_problem_t *original_problem, int **out_row_perm, int **out_col_perm);
static void repermute_solution(cupdlpx_result_t *result, int *row_perm, int *col_perm);

cupdlpx_result_t *distributed_optimize(
    const pdhg_parameters_t *params,
    const lp_problem_t *original_problem
)
{
    pdhg_parameters_t sub_params = *params;

    select_valid_grid_size(params, original_problem, &sub_params);

    grid_context_t grid_context = initialize_parallel_context(
        sub_params.grid_size.row_dims, 
        sub_params.grid_size.col_dims
    );

    sub_params.verbose = (grid_context.rank_global == 0) ? params->verbose : false;
    
    cupdlpx_result_t *result = NULL;
    if (params->permute_method != NO_PERMUTATION)
    {
        lp_problem_t *permuted_problem = NULL;
        int *row_perm = NULL;
        int *col_perm = NULL;
        if (grid_context.rank_global == 0)
        {
            permuted_problem = permute_lp_problem(params, original_problem, &row_perm, &col_perm);
        }

        result =  distributed_optimize_core(&sub_params, permuted_problem, &grid_context);

        if (grid_context.rank_global == 0)
        {
            repermute_solution(result, row_perm, col_perm);
            if (permuted_problem) {
                lp_problem_free(permuted_problem);
            }
            free(row_perm);
            free(col_perm);
        }
    }
    else
    {
        result = distributed_optimize_core(&sub_params, original_problem, &grid_context);
    }
    return result;
}

static cupdlpx_result_t *distributed_optimize_core(const pdhg_parameters_t *params,
                                                   const lp_problem_t *original_problem,
                                                   grid_context_t *grid_context)
{
    print_initial_info(params, original_problem);
    print_distributed_params(params);
    const lp_problem_t *working_problem = original_problem;

    rescale_info_t *rescale_info = NULL;
    cupdlpx_presolve_info_t *presolve_info = NULL;

    int is_solved_during_presolve = 0;

    if (grid_context->rank_global == 0)
    {
        if (params->presolve)
        {
            presolve_info = pslp_presolve(original_problem, params);
            if (presolve_info->problem_solved_during_presolve)
            {
                is_solved_during_presolve = 1;
            }
            else
            {
                working_problem = presolve_info->reduced_problem;
            }
        }

        if (!is_solved_during_presolve)
        {
            rescale_info = rescale_problem(params, working_problem);
        }
    }

    MPI_Bcast(&is_solved_during_presolve, 1, MPI_INT, 0, grid_context->comm_global);

    if (is_solved_during_presolve)
    {
        if (grid_context->rank_global == 0)
        {
            cupdlpx_result_t *result = create_result_from_presolve(presolve_info, original_problem);
            if (presolve_info) cupdlpx_presolve_info_free(presolve_info);
            pdhg_final_log(result, params);
            return result;
        }
        else
        {
            return NULL;
        }
    }

    {
        char *buf = NULL;
        size_t sz = 0;

        if (grid_context->rank_global == 0)
        {
            sz = get_lp_problem_size(working_problem);
            buf = (char *)malloc(sz);
            char *ptr_tmp = buf;
            serialize_lp_problem_to_ptr(working_problem, &ptr_tmp);
        }

        big_bcast_bytes((void **)&buf, &sz, 0, grid_context->comm_global);

        if (grid_context->rank_global != 0)
        {
            const char *ptr_tmp = buf;
            working_problem = deserialize_lp_problem_from_ptr(&ptr_tmp);
        }

        if (buf)
            free(buf);
    }
    {
        char *buf = NULL;
        size_t sz = 0;

        if (grid_context->rank_global == 0)
        {
            sz = get_rescale_info_size(rescale_info);
            buf = (char *)malloc(sz);
            serialize_rescale_info(rescale_info, buf);
        }

        big_bcast_bytes((void **)&buf, &sz, 0, grid_context->comm_global);

        if (grid_context->rank_global != 0)
        {
            rescale_info = deserialize_rescale_info(buf);
        }

        if (buf)
            free(buf);
    }
    int n_start = 0;
    int m_start = 0;
    rescale_info_t *local_rescale_info = partition_rescale_info(
        rescale_info,
        grid_context,
        params->partition_method,
        &n_start,
        &m_start);
    rescale_info_free(rescale_info);
    lp_problem_t *local_working_problem = partition_lp_problem(
        working_problem,
        grid_context,
        params->partition_method,
        &n_start,
        &m_start);
    pdhg_solver_state_t *state = initialize_solver_state(params, local_working_problem, local_rescale_info);
    state->grid_context = grid_context;
    allreduce_obj_bound_norm(state, params);

    rescale_info_free(local_rescale_info);

    initialize_step_size_and_primal_weight_distributed(state, params);

    compute_residual_distributed(state, params->optimality_norm);
    MPI_Barrier(grid_context->comm_global);
    
    double start_time = MPI_Wtime();
    bool do_restart = false;
    while (state->total_count < params->termination_criteria.iteration_limit)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) ||
            (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            compute_residual_distributed(state, params->optimality_norm);
            if (state->is_this_major_iteration &&
                state->total_count < 3 * params->termination_evaluation_frequency)
            {
                compute_infeasibility_information(state);
            }

            state->cumulative_time_sec =
                (double)(MPI_Wtime() - start_time);

            check_termination_criteria(state, &params->termination_criteria);
            display_iteration_stats(state, params->verbose);
            if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED)
            {
                break;
            }
        }

        if ((state->is_this_major_iteration || state->total_count == 0))
        {
            do_restart =
                should_do_adaptive_restart(state, &params->restart_params,
                                           params->termination_evaluation_frequency);
            if (do_restart)
                perform_restart_distributed(state, params);
        }

        state->is_this_major_iteration =
            ((state->total_count + 1) % params->termination_evaluation_frequency) ==
            0;

        compute_next_pdhg_primal_solution_distributed(state);
        compute_next_pdhg_dual_solution_distributed(state);

        if (state->is_this_major_iteration || do_restart)
        {
            compute_fixed_point_error_distributed(state);
            if (do_restart)
            {
                state->initial_fixed_point_error = state->fixed_point_error;
                do_restart = false;
            }
        }

        halpern_update(state, params->reflection_coefficient);

        state->inner_count++;
        state->total_count++;
    }

    if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        compute_residual_distributed(state, params->optimality_norm);
        display_iteration_stats(state, params->verbose);
    }
    cupdlpx_result_t *result = NULL;
    result = create_result_from_state_distributed(state, original_problem);
    if (grid_context->rank_global == 0)
    {
        if (params->presolve && presolve_info)
        {
            pslp_postsolve(presolve_info, result, original_problem);
            cupdlpx_presolve_info_free(presolve_info);
        }
        pdhg_final_log(result, params);
    }
    else
    {
        result = NULL;
    }
    pdhg_solver_state_free(state);
    return result;
}

static void allreduce_obj_bound_norm(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    if (params->optimality_norm == NORM_TYPE_L_INF)
    {
        double local_val = state->objective_vector_norm;
        MPI_Allreduce(&local_val, &state->objective_vector_norm, 1,
                      MPI_DOUBLE, MPI_MAX, state->grid_context->comm_row);
    }
    else
    {
        double local_sq = state->objective_vector_norm * state->objective_vector_norm;
        double global_sq = 0.0;
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM,
                      state->grid_context->comm_row);
        state->objective_vector_norm = sqrt(global_sq);
    }

    if (params->optimality_norm == NORM_TYPE_L_INF)
    {
        double local_val = state->constraint_bound_norm;
        MPI_Allreduce(&local_val, &state->constraint_bound_norm, 1,
                      MPI_DOUBLE, MPI_MAX, state->grid_context->comm_col);
    }
    else
    {
        double local_sq = state->constraint_bound_norm * state->constraint_bound_norm;
        double global_sq = 0.0;
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM,
                      state->grid_context->comm_col);
        state->constraint_bound_norm = sqrt(global_sq);
    }
}

cupdlpx_result_t *create_result_from_state_distributed(pdhg_solver_state_t *state, const lp_problem_t *original_problem)
{
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->pdhg_dual_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

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
        0));

    compute_and_rescale_reduced_cost_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->dual_slack,
        state->objective_vector,
        state->dual_product,
        state->variable_rescaling,
        state->objective_vector_rescaling,
        state->constraint_bound_rescaling,
        state->num_variables);

    rescale_solution(state);

    cupdlpx_result_t *results = NULL;

    if (state->grid_context->rank_global == 0)
    {
        results = (cupdlpx_result_t *)safe_calloc(1, sizeof(cupdlpx_result_t));
    }

    double *global_primal = NULL;
    double *global_dual = NULL;
    double *global_reduced_cost = NULL;

    gather_distributed_vector(
        state->pdhg_primal_solution,
        state->num_variables,
        state->grid_context->comm_col,
        state->grid_context->comm_row,
        &global_primal);

    gather_distributed_vector(
        state->dual_slack,
        state->num_variables,
        state->grid_context->comm_col,
        state->grid_context->comm_row,
        &global_reduced_cost);

    gather_distributed_vector(
        state->pdhg_dual_solution,
        state->num_constraints,
        state->grid_context->comm_row,
        state->grid_context->comm_col,
        &global_dual);

    if (state->grid_context->rank_global == 0)
    {
        if (!global_primal || !global_dual)
        {
            fprintf(stderr, "Error: Failed to gather solution to root.\n");
        }

        results->primal_solution = global_primal;
        results->dual_solution = global_dual;
        results->reduced_cost = global_reduced_cost;

        if (original_problem)
        {
            results->num_variables = original_problem->num_variables;
            results->num_constraints = original_problem->num_constraints;
            results->num_variables = original_problem->num_variables;
            results->num_constraints = original_problem->num_constraints;
            results->num_nonzeros = original_problem->constraint_matrix_num_nonzeros;
        }
        results->total_count = state->total_count;
        results->rescaling_time_sec = state->rescaling_time_sec;
        results->cumulative_time_sec = state->cumulative_time_sec;
        results->relative_primal_residual = state->relative_primal_residual;
        results->relative_dual_residual = state->relative_dual_residual;
        results->absolute_primal_residual = state->absolute_primal_residual;
        results->absolute_dual_residual = state->absolute_dual_residual;
        results->primal_objective_value = state->primal_objective_value;
        results->dual_objective_value = state->dual_objective_value;
        results->objective_gap = state->objective_gap;
        results->relative_objective_gap = state->relative_objective_gap;
        results->max_primal_ray_infeasibility = state->max_primal_ray_infeasibility;
        results->max_dual_ray_infeasibility = state->max_dual_ray_infeasibility;
        results->primal_ray_linear_objective = state->primal_ray_linear_objective;
        results->dual_ray_objective = state->dual_ray_objective;
        results->termination_reason = state->termination_reason;
        results->feasibility_polishing_time = state->feasibility_polishing_time;
        results->feasibility_iteration = state->feasibility_iteration;
    }

    return results;
}

void get_best_grid_dims(int m, int n, int n_procs, int *out_r, int *out_c)
{
    int best_r = 1;
    int best_c = n_procs;
    double best_score = DBL_MAX;

    if (n == 0)
    {
        *out_r = best_r;
        *out_c = best_c;
        return;
    }

    double target_ratio = (double)m / (double)n;

    for (int r = 1; r <= n_procs; ++r)
    {
        if (n_procs % r == 0)
        {
            int c = n_procs / r;

            double grid_ratio = (double)r / (double)c;
            double score = fabs(log(grid_ratio) - log(target_ratio));

            if (score < best_score)
            {
                best_score = score;
                best_r = r;
                best_c = c;
            }
        }
    }

    *out_r = best_r;
    *out_c = best_c;
}

static void select_valid_grid_size(const pdhg_parameters_t *params, const lp_problem_t *original_problem, pdhg_parameters_t *sub_params)
{
    int world_size, rank_global;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_global);

    if (params->grid_size.decided) 
    {
        int provided_rows = params->grid_size.row_dims;
        int provided_cols = params->grid_size.col_dims;
        int product = provided_rows * provided_cols;

        if (product != world_size) 
        {
            if (rank_global == 0) 
            {
                fprintf(stderr, "\n[Error] MPI World Size Mismatch!\n");
                fprintf(stderr, "------------------------------------------------\n");
                fprintf(stderr, "User specified grid:  %d x %d = %d processes\n", provided_rows, provided_cols, product);
                fprintf(stderr, "Actual MPI world size: %d processes\n", world_size);
                fprintf(stderr, "Please adjust -n (mpirun) or --n_row_tiles/--n_col_tiles.\n");
                fprintf(stderr, "------------------------------------------------\n");
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        sub_params->grid_size.row_dims = provided_rows;
        sub_params->grid_size.col_dims = provided_cols;
    }
    else 
    {
        int dims[2]; 

        if (rank_global == 0) 
        {
            get_best_grid_dims(
                original_problem->num_constraints, 
                original_problem->num_variables, 
                world_size, 
                &dims[0], 
                &dims[1]
            );
            
            if (params->verbose) {
                printf("[Auto-Grid] Decided grid shape: %d x %d for %d processes.\n", dims[0], dims[1], world_size);
            }
        }

        MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);

        sub_params->grid_size.row_dims = dims[0];
        sub_params->grid_size.col_dims = dims[1];
        sub_params->grid_size.decided = 1; 
    }
    return;
}

static lp_problem_t *permute_lp_problem(const pdhg_parameters_t *params, const lp_problem_t *original_problem, int **out_row_perm, int **out_col_perm)
{
    *out_row_perm = (int *)malloc(original_problem->num_constraints * sizeof(int));
    *out_col_perm = (int *)malloc(original_problem->num_variables * sizeof(int));
    
    int *row_perm = *out_row_perm;
    int *col_perm = *out_col_perm;

    if (params->permute_method == FULL_RANDOM_PERMUTATION)
    {
        generate_random_permutation(original_problem->num_variables, col_perm);
        generate_random_permutation(original_problem->num_constraints, row_perm);
    }
    else if (params->permute_method == BLOCK_RANDOM_PERMUTATION)
    {
        generate_block_permutation(original_problem->num_variables, 128, col_perm);
        generate_block_permutation(original_problem->num_constraints, 128, row_perm);
    }

    lp_problem_t *new_problem = permute_problem_return_new(original_problem, row_perm, col_perm);
    return new_problem;
}

static void repermute_solution(cupdlpx_result_t *result, int *row_perm, int *col_perm)
{
    int *inv_col_perm = (int *)malloc(result->num_variables * sizeof(int));
    int *inv_row_perm = (int *)malloc(result->num_constraints * sizeof(int));
    compute_inv_perm(result->num_variables, col_perm, inv_col_perm);
    compute_inv_perm(result->num_constraints, row_perm, inv_row_perm);
    permute_double_array(result->primal_solution, result->num_variables, inv_col_perm);
    permute_double_array(result->dual_solution, result->num_constraints, inv_row_perm);
    permute_double_array(result->reduced_cost, result->num_variables, inv_col_perm);
    free(inv_col_perm);
    free(inv_row_perm);
}