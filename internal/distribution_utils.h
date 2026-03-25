#pragma once
#include "internal_types.h"
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes the MPI and NCCL contexts for a 2D processor grid.
 * * @param P_row Number of processor rows
 * @param P_col Number of processor columns
 * @return GridContext The initialized context structure
 */

grid_context_t initialize_parallel_context(int P_row, int P_col);
rescale_info_t *partition_rescale_info(rescale_info_t *global_info,
                                       const grid_context_t *grid,
                                       partition_method_t method,
                                       int *out_n_start, int *out_m_start);
lp_problem_t *partition_lp_problem(const lp_problem_t *global_lp,
                                   const grid_context_t *grid,
                                   partition_method_t method, int *out_n_start,
                                   int *out_m_start);

rescale_info_t *deserialize_rescale_info(const char *buffer);
void serialize_rescale_info(const rescale_info_t *info, char *buffer);
size_t get_rescale_info_size(const rescale_info_t *info);
lp_problem_t *deserialize_lp_problem_from_ptr(const char **ptr_ref);
void serialize_lp_problem_to_ptr(const lp_problem_t *lp, char **ptr_ref);
size_t get_lp_problem_size(const lp_problem_t *lp);
void big_bcast_bytes(void **buffer_ptr, size_t *size_ptr, int root,
                     MPI_Comm comm);
void big_send_bytes(const void *buffer, size_t size, int dest, MPI_Comm comm);
void big_recv_bytes(void **buffer_ptr, size_t *size_ptr, int source,
                    MPI_Comm comm);
big_request_t big_isend_bytes(const void *buffer, size_t size, int dest,
                              MPI_Comm comm);
void big_wait_bytes(big_request_t *breq, unsigned long long *p_len);
void distribute_data_bcast_then_partition(const lp_problem_t *working_problem,
                                          rescale_info_t *rescale_info,
                                          grid_context_t *grid_context,
                                          const pdhg_parameters_t *params,
                                          lp_problem_t **out_local_lp,
                                          rescale_info_t **out_local_resc);
void distribute_data_partition_then_send(const lp_problem_t *working_problem,
                                         rescale_info_t *rescale_info,
                                         grid_context_t *grid_context,
                                         const pdhg_parameters_t *params,
                                         lp_problem_t **out_local_lp,
                                         rescale_info_t **out_local_resc);
void initialize_step_size_and_primal_weight_distributed(
    pdhg_solver_state_t *state, const pdhg_parameters_t *params);
void gather_distributed_vector(double *d_local_vec, int local_len,
                               MPI_Comm comm_check, MPI_Comm comm_gather,
                               double **result_ptr);
void print_distributed_params(const pdhg_parameters_t *params);
#ifdef __cplusplus
}
#endif
