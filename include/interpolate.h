#ifndef interpolate_h
#define interpolate_h

#include <math.h>
#include <stdbool.h>

#include "memory.h"

#ifdef __NVCC__

#define DEF_UNROLL_SIZE 256

typedef struct velo_grid {
  float *x;
  float *y;
  float *val;
} velo_grid;
typedef struct mass_grid {
  float *x;
  float *y;
  float *u;
  float *v;
  float *w;
  float *ph;
  float *phb;
} mass_grid;

__global__ void gpu_radially_interpolate_unstructured(velo_grid *u_grid, velo_grid *v_grid, mass_grid *m_grid,
   const int NY_STAG, const int NX_STAG, const int NY, const int NX, const int z, const int dim,
   const int num_support_points, const float exponent);

__global__ void gpu_radially_interpolate_structured_horiz(velo_grid *u_grid, velo_grid *v_grid, mass_grid *m_grid,
   const int NY, const int NX, const int num_support_points, const float exponent);

__global__ void gpu_radially_interpolate_structured_vert(velo_grid *w_grid, mass_grid *m_grid,
      const int NY, const int NX, float z_level, float z_level_stag_under, float z_level_stag_above,
      const int num_support_points, const float exponent);

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

float cpu_radially_interpolate_structured_horiz(velo_grid *velo_grid, float *xi, float *yi,
            int idx, const int NY, const int NX, const int num_support_points, const float exponent);

float cpu_radially_interpolate_structured_vert(velo_grid *velo_grid, float *xi, float *yi,
            int idx, const int NY, const int NX, const int num_support_points, const float exponent);

#endif
