
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NROWS_OUTPUT 4000
#define NCOLS_OUTPUT 4000
#define INNER_DIM 4000

#define NROWS_A NROWS_OUTPUT
#define NCOLS_A INNER_DIM

#define NROWS_B INNER_DIM
#define NCOLS_B NCOLS_OUTPUT
float float_rand(float min, float max) {
  float scale = rand() / (float)RAND_MAX; /* [0, 1.0] */
  return min + scale * (max - min);       /* [min, max] */
}
#include <sys/time.h>

long long current_timestamp() {
  struct timeval te;
  gettimeofday(&te, NULL); // get current time
  long long milliseconds =
      te.tv_sec * 1000LL + te.tv_usec / 1000; // calculate milliseconds
  // printf("milliseconds: %lld\n", milliseconds);
  return milliseconds;
}

int main() {
  srand(0);
  int i = 0, j = 0;
  float *A = malloc(NROWS_A * NCOLS_A * sizeof(float));
  float *B = malloc(NROWS_B * NCOLS_B * sizeof(float));
  float *C = malloc(NROWS_OUTPUT * NCOLS_OUTPUT * sizeof(float));
  for (i = 0; i < NROWS_A; i++)
    for (j = 0; j < NCOLS_A; j++) {
      A[i * NROWS_A + j] = float_rand(0.0, 1.0);
      B[i * NROWS_B + j] = float_rand(0.0, 1.0);
    }

  long before_time = current_timestamp();
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NROWS_OUTPUT,
              NCOLS_OUTPUT, INNER_DIM, 1.0, A, NROWS_A, B, NROWS_B, 1.0, C,
              NROWS_OUTPUT);
  long after_time = current_timestamp();
  printf("Took: %lf s\n", (after_time - before_time) / (double)1000.0);
  // for (i = 0; i < NROWS_OUTPUT; i++, puts(""))
  //   for (j = 0; j < NCOLS_OUTPUT; j++)
  //     printf("%f ", C[i * NROWS_OUTPUT + j]);
  return 0;
}
