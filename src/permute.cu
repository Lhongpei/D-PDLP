#include "permute.h"
#include <math.h>
#include <random>
#include <algorithm>
#include <vector>
#include <omp.h>

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

struct permute_tuple_t {
  int new_col;
  double val;
  bool operator<(const permute_tuple_t& other) const {
    return new_col < other.new_col;
  }
};

int cmp_tuples(const void *a, const void *b) {
  return ((permute_tuple_t *)a)->new_col - ((permute_tuple_t *)b)->new_col;
}

void col_permute_in_place(int m, int *Ap, int *Aj, double *Ax,
                          const int *old_col_to_new) {
  #pragma omp parallel for schedule(dynamic, 1024)
  for (int r = 0; r < m; r++) {
    int start = Ap[r];
    int end = Ap[r + 1];
    int len = end - start;

    if (len <= 1)
      continue;

    std::vector<permute_tuple_t> local_buffer(len);

    for (int k = 0; k < len; k++) {
      int current_idx = start + k;
      int old_col = Aj[current_idx];

      local_buffer[k].new_col = old_col_to_new[old_col];
      local_buffer[k].val = Ax[current_idx];
    }

    std::sort(local_buffer.begin(), local_buffer.end());

    for (int k = 0; k < len; k++) {
      int current_idx = start + k;
      Aj[current_idx] = local_buffer[k].new_col;
      Ax[current_idx] = local_buffer[k].val;
    }
  }
}

void permute_rows_structural(lp_problem_t *qp, const int *row_perm) {
  int m = qp->num_constraints;
  int nnz = qp->constraint_matrix_num_nonzeros;

  int *new_Ap = (int *)malloc((m + 1) * sizeof(int));
  int *new_Aj = (int *)malloc(nnz * sizeof(int));
  double *new_Ax = (double *)malloc(nnz * sizeof(double));

  new_Ap[0] = 0;

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < m; i++) {
    int old_row_idx = row_perm[i];
    int len = qp->constraint_matrix_row_pointers[old_row_idx + 1] - 
              qp->constraint_matrix_row_pointers[old_row_idx];
    new_Ap[i + 1] = len;
  }

  for (int i = 0; i < m; i++) {
    new_Ap[i + 1] += new_Ap[i];
  }

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < m; i++) {
    int old_row_idx = row_perm[i];
    int start = qp->constraint_matrix_row_pointers[old_row_idx];
    int len = new_Ap[i + 1] - new_Ap[i];
    int dest_offset = new_Ap[i];

    if (len > 0) {
      memcpy(&new_Aj[dest_offset], &qp->constraint_matrix_col_indices[start],
             len * sizeof(int));
      memcpy(&new_Ax[dest_offset], &qp->constraint_matrix_values[start],
             len * sizeof(double));
    }
  }

  free(qp->constraint_matrix_row_pointers);
  free(qp->constraint_matrix_col_indices);
  free(qp->constraint_matrix_values);

  qp->constraint_matrix_row_pointers = new_Ap;
  qp->constraint_matrix_col_indices = new_Aj;
  qp->constraint_matrix_values = new_Ax;
}

void permute_double_array(double *arr, int n, const int *perm) {
  if (!arr)
    return;
  double *tmp = (double *)malloc(n * sizeof(double));
  
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++)
    tmp[i] = arr[perm[i]];
    
  memcpy(arr, tmp, n * sizeof(double));
  free(tmp);
}

void compute_inv_perm(int n, const int *perm, int *inv_perm) {
  for (int i = 0; i < n; i++)
    inv_perm[perm[i]] = i;
}

void permute_problem(lp_problem_t *qp, int *row_perm, int *col_perm) {
  int m = qp->num_constraints;
  int n = qp->num_variables;

  permute_double_array(qp->objective_vector, n, col_perm);
  permute_double_array(qp->variable_lower_bound, n, col_perm);
  permute_double_array(qp->variable_upper_bound, n, col_perm);
  permute_double_array(qp->primal_start, n, col_perm);

  permute_double_array(qp->constraint_lower_bound, m, row_perm);
  permute_double_array(qp->constraint_upper_bound, m, row_perm);
  permute_double_array(qp->dual_start, m, row_perm);

  permute_rows_structural(qp, row_perm);

  int *inv_col_perm = (int *)malloc(n * sizeof(int));
  compute_inv_perm(n, col_perm, inv_col_perm);

  col_permute_in_place(m, qp->constraint_matrix_row_pointers,
                       qp->constraint_matrix_col_indices,
                       qp->constraint_matrix_values, inv_col_perm);

  free(inv_col_perm);
}

lp_problem_t *permute_problem_return_new(const lp_problem_t *qp, int *row_perm,
                                         int *col_perm) {
  if (!qp)
    return NULL;

  lp_problem_t *new_qp = (lp_problem_t *)malloc(sizeof(lp_problem_t));
  if (!new_qp)
    return NULL;

  new_qp->num_variables = qp->num_variables;
  new_qp->num_constraints = qp->num_constraints;
  new_qp->constraint_matrix_num_nonzeros = qp->constraint_matrix_num_nonzeros;
  new_qp->objective_constant = qp->objective_constant;
  
  int n = qp->num_variables;
  int m = qp->num_constraints;
  int nnz = qp->constraint_matrix_num_nonzeros;

#define DEEP_COPY_ARRAY(dest, src, count, type)                                \
  if (src) {                                                                   \
    dest = (type *)malloc((count) * sizeof(type));                             \
    memcpy(dest, src, (count) * sizeof(type));                                 \
  } else {                                                                     \
    dest = NULL;                                                               \
  }

  DEEP_COPY_ARRAY(new_qp->objective_vector, qp->objective_vector, n, double);
  DEEP_COPY_ARRAY(new_qp->variable_lower_bound, qp->variable_lower_bound, n,
                  double);
  DEEP_COPY_ARRAY(new_qp->variable_upper_bound, qp->variable_upper_bound, n,
                  double);
  DEEP_COPY_ARRAY(new_qp->primal_start, qp->primal_start, n, double);

  DEEP_COPY_ARRAY(new_qp->constraint_lower_bound, qp->constraint_lower_bound, m,
                  double);
  DEEP_COPY_ARRAY(new_qp->constraint_upper_bound, qp->constraint_upper_bound, m,
                  double);
  DEEP_COPY_ARRAY(new_qp->dual_start, qp->dual_start, m, double);

  DEEP_COPY_ARRAY(new_qp->constraint_matrix_row_pointers,
                  qp->constraint_matrix_row_pointers, m + 1, int);
  DEEP_COPY_ARRAY(new_qp->constraint_matrix_col_indices,
                  qp->constraint_matrix_col_indices, nnz, int);
  DEEP_COPY_ARRAY(new_qp->constraint_matrix_values,
                  qp->constraint_matrix_values, nnz, double);

#undef DEEP_COPY_ARRAY
  permute_problem(new_qp, row_perm, col_perm);

  return new_qp;
}

void generate_random_permutation(int n, int *perm) {
  for (int i = 0; i < n; i++)
    perm[i] = i;
  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    int t = perm[i];
    perm[i] = perm[j];
    perm[j] = t;
  }
}

void randomly_permute_problem(lp_problem_t *qp, int **out_row_perm,
                              int **out_col_perm) {
  int m = qp->num_constraints;
  int n = qp->num_variables;

  int *row_perm = (int *)malloc(m * sizeof(int));
  int *col_perm = (int *)malloc(n * sizeof(int));

  generate_random_permutation(m, row_perm);
  generate_random_permutation(n, col_perm);

  permute_problem(qp, row_perm, col_perm);

  *out_row_perm = row_perm;
  *out_col_perm = col_perm;
}

void generate_block_permutation(int n, int block_size, int *perm) {
  if (block_size <= 0)
    block_size = 1;
  int num_blocks = (n + block_size - 1) / block_size;

  int *block_indices = (int *)malloc(num_blocks * sizeof(int));
  if (!block_indices)
    return;

  for (int i = 0; i < num_blocks; i++) {
    block_indices[i] = i;
  }

  for (int i = num_blocks - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    int temp = block_indices[i];
    block_indices[i] = block_indices[j];
    block_indices[j] = temp;
  }

  int current_pos = 0;

  for (int i = 0; i < num_blocks; i++) {
    int b_idx = block_indices[i];

    int start_val = b_idx * block_size;
    int end_val = MIN((b_idx + 1) * block_size, n);

    for (int val = start_val; val < end_val; val++) {
      perm[current_pos++] = val;
    }
  }
  free(block_indices);
}

void randomly_block_permute_problem(lp_problem_t *qp, int row_block_size,
                                    int col_block_size, int **out_row_perm,
                                    int **out_col_perm) {
  int m = qp->num_constraints;
  int n = qp->num_variables;

  int *row_perm = (int *)malloc(m * sizeof(int));
  int *col_perm = (int *)malloc(n * sizeof(int));

  generate_block_permutation(m, row_block_size, row_perm);
  generate_block_permutation(n, col_block_size, col_perm);

  permute_problem(qp, row_perm, col_perm);

  *out_row_perm = row_perm;
  *out_col_perm = col_perm;
}
