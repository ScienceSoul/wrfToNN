#include <math.h>
#include "finite_difference.h"

void get_stencils_values(fd_container *fd_container, float *ddx, float *ddy, float dx, float dy, 
  int NY, int NX, int NY_STAG, int NX_STAG, int z) {
    
    for(int j=0; j<NY; j++) {
      int jp1 = minf(j+1, NY-1);
      int jm1 = maxf(j-1, 0);
      for(int i=0; i<NX; i++) {
        int ip1 = minf(i+1, NX-1);
        int im1 = maxf(i-1, 0);

        float dsx = (ip1-im1)*dx;
        float dsy = (jp1-jm1)*dy;

        // Store the stencils for dv/dx
        fd_container->stencils_val[((j*NX)+i)*10] = dsx;
        fd_container->stencils_val[(((j*NX)+i)*10)+1] = ddx[(z*(NY_STAG*NX))+((j*NX)+ip1)];
        fd_container->stencils_val[(((j*NX)+i)*10)+2] = ddx[(z*(NY_STAG*NX))+(((j+1)*NX)+ip1)];
        fd_container->stencils_val[(((j*NX)+i)*10)+3] = ddx[(z*(NY_STAG*NX))+((j*NX)+im1)];
        fd_container->stencils_val[(((j*NX)+i)*10)+4] = ddx[(z*(NY_STAG*NX))+(((j+1)*NX)+im1)];

        // Store the stencils for du/dy
        fd_container->stencils_val[(((j*NX)+i)*10)+5] = dsy;
        fd_container->stencils_val[(((j*NX)+i)*10)+6] = ddy[(z*(NY*NX_STAG))+((jp1*NX_STAG)+i)];
        fd_container->stencils_val[(((j*NX)+i)*10)+7] = ddy[(z*(NY*NX_STAG))+((jp1*NX_STAG)+i+1)];
        fd_container->stencils_val[(((j*NX)+i)*10)+8] = ddy[(z*(NY*NX_STAG))+((jm1*NX_STAG)+i)];
        fd_container->stencils_val[(((j*NX)+i)*10)+9] = ddy[(z*(NY*NX_STAG))+((jm1*NX_STAG)+i+1)];       
      }
    }
  }

#ifdef __NVCC__
__device__ void get_rel_vert_vort(fd_container *fd_container, int idx, float *dv_dx, float *du_dy) {

   float dx   = fd_container->stencils_val[(idx*10)];
   float v1_1 = fd_container->stencils_val[(idx*10)+1];
   float v1_2 = fd_container->stencils_val[(idx*10)+2];
   float v2_1 = fd_container->stencils_val[(idx*10)+3];
   float v2_2 = fd_container->stencils_val[(idx*10)+4];

   float dy   = fd_container->stencils_val[(idx*10)+5];
   float u1_1 = fd_container->stencils_val[(idx*10)+6];
   float u1_2 = fd_container->stencils_val[(idx*10)+7];
   float u2_1 = fd_container->stencils_val[(idx*10)+8];
   float u2_2 = fd_container->stencils_val[(idx*10)+9];

   *dv_dx = 0.5f * (v1_1 + v1_2 - v2_1 - v2_2) / dx;
   *du_dy = 0.5f * (u1_1 + u1_2 - u2_1 - u2_2) / dy;
}

__global__ void gpu_compute_rel_vert_vort(fd_container *fd_container, const int NY, const int NX, 
   float scaling_factor) {

   int IDX = blockIdx.x * blockDim.x + threadIdx.x;

   if (IDX >= (NY*NX)) return;

   float dv_dx = 0.0f;
   float du_dy = 0.0f;
   get_rel_vert_vort(fd_container, IDX, &dv_dx, &du_dy);

   fd_container->val[IDX] = (dv_dx - du_dy)*scaling_factor;
}

__global__ void gpu_compute_abs_vert_vort(fd_container *fd_container, const int NY, const int NX, 
   float scaling_factor) {

   int IDX = blockIdx.x * blockDim.x + threadIdx.x;

   if (IDX >= (NY*NX)) return;

   float dv_dx = 0.0f;
   float du_dy = 0.0f;
   get_rel_vert_vort(fd_container, IDX, &dv_dx, &du_dy);

   float earth_angular_velocity = 7.2921e-5; // rad/s
   float rad_lat = fd_container->buffer[IDX] * M_PI/180.0f;
   float f = 2.0f*earth_angular_velocity*sinf(rad_lat);

   fd_container->val[IDX] = (f + (dv_dx - du_dy))*scaling_factor;
 }
#endif
