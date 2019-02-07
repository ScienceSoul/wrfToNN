#ifndef interpolate_h
#define interpolate_h

#include <math.h>
#include <stdbool.h>

#include "memory.h"

#ifdef __NVCC__

#define UNROLL_SIZE 256

typedef struct velo_grid {
  float *x;
  float *y;
  float *z;
  float *val;
} velo_grid;
typedef struct mass_grid {
  float *x;
  float *y;
  float *z;
  float *u;
  float *v;
} mass_grid;

__global__ void radially_interpolate_gpu(velo_grid *u_grid, velo_grid *v_grid, mass_grid *m_grid,
   const int NY_STAG, const int NX_STAG, const int NY, const int NX, const int z, const int dim,
   const int num_support_points, const float exponent);

#endif


float radially_interpolate_cpu(float **data,
                           float *xi,
                           float *yi,
                           float *zi,
                           int num_data,
                           int dim,
                           int *directions,
                           float exponent,
                           bool reinitiate,
                           int num_support_points,
                           bool *verbose);

#endif
