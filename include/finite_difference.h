#ifndef fd_h
#define fd_h

#include "memory.h"

typedef struct fd_container {
  float *stencils_val;  // Store the values at the stencils used to
                        // compute derivative at a given node
  float *val;           // The container contains two buffers for results
  float *buffer;        // A buffer needed to store an auxilarry field
} fd_container;

//void get_stencils_values(fd_tags *fd_tags, fd_container *fd_container, float *ddx, float *ddy, int NY, int NX,
//     int z);

void get_stencils_values(fd_container *fd_container, float *ddx, float *ddy, float dx, float dy, 
     int NY, int NX, int NY_STAG, int NX_STAG, int z);

#ifdef __NVCC__
__global__ void gpu_compute_rel_vert_vort(fd_container *fd_container, const int NY, const int NX, float scaling_factor);
__global__ void gpu_compute_abs_vert_vort(fd_container *fd_container, const int NY, const int NX, float scaling_factor);
#endif

#endif
