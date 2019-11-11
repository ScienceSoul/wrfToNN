#include <math.h>
#include "finite_difference.h"

fd_tags *allocate_fd_tags(int n) {

  fd_tags *tags = (fd_tags *)malloc(n*sizeof(fd_tags));
  for (int i = 0; i < n; i++) {
    tags[i].ddx = -1;
    tags[i].ddy = -1;
  }
  return tags;
}

void set_fd_tags(fd_tags *fd_tags, int NY, int NX) {

  int k = 0;
  for (int j = 0; j < NY; j++) {
    for (int i = 0; i < NX; i++) {

      if (j == 0 && i == 0) { // Lower left node
        fd_tags[k].ddx = FORWARD_DIFF;
        fd_tags[k].ddy = FORWARD_DIFF;
        k++;
        continue;
      }

      if (j == 0) { // Lower side
        if (i == NX-1) { // Lower right node
          fd_tags[k].ddx = BACKWARD_DIFF;
          fd_tags[k].ddy = FORWARD_DIFF;
        } else {
          fd_tags[k].ddx = CENTRAL_DIFF;
          fd_tags[k].ddy = FORWARD_DIFF;
        }
        k++;
        continue;
      }

      if (i == 0) { // Left side
        if (j == NY-1) { // Upper left node
          fd_tags[k].ddx = FORWARD_DIFF;
          fd_tags[k].ddy = BACKWARD_DIFF;
        } else {
          fd_tags[k].ddx = FORWARD_DIFF;
          fd_tags[k].ddy = CENTRAL_DIFF;
        }
        k++;
        continue;
      }

      if (j == NY-1) { // Upper side
        if (i == NX-1) { // Upper right node
          fd_tags[k].ddx = BACKWARD_DIFF;
          fd_tags[k].ddy = BACKWARD_DIFF;
        } else {
          fd_tags[k].ddx = CENTRAL_DIFF;
          fd_tags[k].ddy = BACKWARD_DIFF;
        }
        k++;
        continue;
      }

      if (i == NX-1) { // Right side
        fd_tags[k].ddx = BACKWARD_DIFF;
        fd_tags[k].ddy = CENTRAL_DIFF;
        k++;
        continue;
      }

      // Inner domain
      fd_tags[k].ddx = CENTRAL_DIFF;
      fd_tags[k].ddy = CENTRAL_DIFF;
      k++;
    }
  }
}

void get_stencils_values(fd_tags *fd_tags, fd_container *fd_container, float *ddx, float *ddy, int NY, int NX,
     int z) {

  int k = 0;
  for (int y = 0; y < NY; y++) {
    for (int x = 0; x < NX; x++) {
      switch (fd_tags[k].ddx) {
        case FORWARD_DIFF:
          fd_container->stencils_val[((y*NX)+x)*6] = (float)FORWARD_DIFF;
          fd_container->stencils_val[(((y*NX)+x)*6)+1] = ddx[(z*(NY*NX))+((y*NX)+x)+1];
          fd_container->stencils_val[(((y*NX)+x)*6)+2] = ddx[(z*(NY*NX))+((y*NX)+x)];
          break;
        case BACKWARD_DIFF:
          fd_container->stencils_val[((y*NX)+x)*6] = (float)BACKWARD_DIFF;
          fd_container->stencils_val[(((y*NX)+x)*6)+1] = ddx[(z*(NY*NX))+((y*NX)+x)];
          fd_container->stencils_val[(((y*NX)+x)*6)+2] = ddx[(z*(NY*NX))+((y*NX)+x)-1];
          break;
        case CENTRAL_DIFF:
          fd_container->stencils_val[((y*NX)+x)*6] = (float)CENTRAL_DIFF;
          fd_container->stencils_val[(((y*NX)+x)*6)+1] = ddx[(z*(NY*NX))+((y*NX)+x)+1];
          fd_container->stencils_val[(((y*NX)+x)*6)+2] = ddx[(z*(NY*NX))+((y*NX)+x)-1];
          break;
      }

      switch (fd_tags[k].ddy) {
        case FORWARD_DIFF:
          fd_container->stencils_val[(((y*NX)+x)*6)+3] = (float)FORWARD_DIFF;
          fd_container->stencils_val[(((y*NX)+x)*6)+4] = ddy[(z*(NY*NX))+(((y+1)*NX)+x)];
          fd_container->stencils_val[(((y*NX)+x)*6)+5] = ddy[(z*(NY*NX))+((y*NX)+x)];
          break;
        case BACKWARD_DIFF:
          fd_container->stencils_val[(((y*NX)+x)*6)+3] = (float)BACKWARD_DIFF;
          fd_container->stencils_val[(((y*NX)+x)*6)+4] = ddy[(z*(NY*NX))+((y*NX)+x)];
          fd_container->stencils_val[(((y*NX)+x)*6)+5] = ddy[(z*(NY*NX))+(((y-1)*NX)+x)];
          break;
        case CENTRAL_DIFF:
          fd_container->stencils_val[(((y*NX)+x)*6)+3] = (float)CENTRAL_DIFF;
          fd_container->stencils_val[(((y*NX)+x)*6)+4] = ddy[(z*(NY*NX))+(((y+1)*NX)+x)];
          fd_container->stencils_val[(((y*NX)+x)*6)+5] = ddy[(z*(NY*NX))+(((y-1)*NX)+x)];
          break;
      }
      k++;
    }
  }
}

#ifdef __NVCC__
__device__ void get_rel_vert_vort(fd_container *fd_container, int idx, float *dv_dx, float *du_dy, int dy, int dx) {

   float ddx_flag = fd_container->stencils_val[(idx*6)];
   float v1       = fd_container->stencils_val[(idx*6)+1];
   float v2       = fd_container->stencils_val[(idx*6)+2];
   float ddy_flag = fd_container->stencils_val[(idx*6)+3];
   float u1       = fd_container->stencils_val[(idx*6)+4];
   float u2       = fd_container->stencils_val[(idx*6)+5];

   *dv_dx = (ddx_flag == 2.0f) ? (v1-v2)/(2.0f*dx) : (v1-v2)/dx;
   *du_dy = (ddy_flag == 2.0f) ? (u1-u2)/(2.0f*dy) : (u1-u2)/dy;
}

__global__ void gpu_compute_rel_vert_vort(fd_container *fd_container, const int NY, const int NX, const int dy,
   const int dx) {

   int IDX = blockIdx.x * blockDim.x + threadIdx.x;

   if (IDX >= (NY*NX)) return;

   float dv_dx = 0.0f;
   float du_dy = 0.0f;
   get_rel_vert_vort(fd_container, IDX, &dv_dx, &du_dy, dy, dx);

   fd_container->val[IDX] = (dv_dx - du_dy)*1.0e06;
}

__global__ void gpu_compute_abs_vert_vort(fd_container *fd_container, const int NY, const int NX, const int dy,
   const int dx) {

   int IDX = blockIdx.x * blockDim.x + threadIdx.x;

   if (IDX >= (NY*NX)) return;

   float dv_dx = 0.0f;
   float du_dy = 0.0f;
   get_rel_vert_vort(fd_container, IDX, &dv_dx, &du_dy, dy, dx);

   float earth_angular_velocity = 7.2921e-5; // rad/s
   float rad_lat = fd_container->buffer[IDX] * M_PI/180.0f;
   float f = 2.0f*earth_angular_velocity*sinf(rad_lat);

   fd_container->val[IDX] = (f + (dv_dx - du_dy))*1.0e06; // Return the result in (micoseconds)^-1
 }
#endif
