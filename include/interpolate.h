#ifndef interpolate_h
#define interpolate_h

#include <math.h>
#include <stdbool.h>

#include "memory.h"

#ifdef __NVCC__

#define UNROLL_SIZE 256
#define BLOCK_SIZE 128

enum {
  STRUCTURED=1,
  UNSTRUCTURED
};

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

__global__ void gpu_radially_interpolate_unstructured(velo_grid *u_grid, velo_grid *v_grid, mass_grid *m_grid,
   const int NY_STAG, const int NX_STAG, const int NY, const int NX, const int z, const int dim,
   const int num_support_points, const float exponent);

__global__ void gpu_radially_interpolate_structured(velo_grid *u_grid, velo_grid *v_grid, mass_grid *m_grid,
   const int NY, const int NX, const int num_support_points, const float exponent);

#endif

float cpu_radially_interpolate_unstructured(float **data,
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

float cpu_radially_interpolate_structured(velo_grid *velo_grid, float *xi, float *yi,
            int idx, const int NY, const int NX, const int num_support_points, const float exponent);

#endif
