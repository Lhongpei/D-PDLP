#pragma once

#include "cupdlpx_types.h"
#include "cupdlpx.h"
#include "internal_types.h"
#include "preconditioner.h"
#include "presolve.h"
#include "solver.h"
#include "utils.h"

#ifdef __cplusplus
extern "C"
{
#endif
    #define NCCL_CHECK(cmd) do {                         \
    ncclResult_t r = cmd;                              \
    if (r != ncclSuccess) {                            \
        printf("NCCL failure %s:%d '%s'\n",              \
            __FILE__, __LINE__, ncclGetErrorString(r));  \
        exit(EXIT_FAILURE);                              \
    }                                                  \
    } while(0)
    void compute_next_pdhg_primal_solution_distributed(pdhg_solver_state_t *state);
    void compute_next_pdhg_dual_solution_distributed(pdhg_solver_state_t *state);
    void perform_restart_distributed(pdhg_solver_state_t *state, const pdhg_parameters_t *params);
    void compute_fixed_point_error_distributed(pdhg_solver_state_t *state);
    void compute_residual_distributed(pdhg_solver_state_t *state, norm_type_t optimality_norm);
#ifdef __cplusplus
}
#endif