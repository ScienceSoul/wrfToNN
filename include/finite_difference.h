#ifndef fd_h
#define fd_h

#include "memory.h"

typedef enum fd_tag {
  FORWARD_DIFF=0,
  BACKWARD_DIFF,
  CENTRAL_DIFF
} fd_tag;

typedef struct fd_tags {  // Store the fd tags for each node of the plane domain
  int ddx;
  int ddy;
} fd_tags;

typedef struct fd_container {
  float *stencils_val;  // Store the values at the stencils used to
                        // compute derivative at a given node
  float *val;          // The container contains two buffers for results
  float *buffer;        // A buffer needed to store an auxilarry field
} fd_container;

fd_tags *allocate_fd_tags(int n);
int set_fd_tags(fd_tags *fd_tags, int NY, int NX);

void get_stencils_values(fd_tags *fd_tags, fd_container *fd_container, float *ddx, float *ddy, int NY, int NX,
     int z);

#ifdef __NVCC__
__global__ void gpu_compute_rel_vert_vort(fd_container *fd_container, const int NY, const int NX, const int dy,
   const int dx, float scaling_factor);
__global__ void gpu_compute_abs_vert_vort(fd_container *fd_container, const int NY, const int NX, const int dy,
   const int dx, float scaling_factor);
#endif

#endif
