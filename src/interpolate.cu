#include "interpolate.h"

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
                               bool *verbose) {

    float interpolated_value;
    float x[3], minmaxxy[2][3];
    static float max_distance = 0.0f;

    if (num_support_points < 2) {
      fprintf(stderr, "The number of points must be at least 2.\n");
      exit(EXIT_FAILURE);
    }
    if (exponent < 1) {
      fprintf(stderr, "The exponent should be larger or equal to unity.\n");
      exit(EXIT_FAILURE);
    }

    if (verbose != NULL) {
      static int count = 0;
      if (*verbose) {
        count++;
        if (dim > 2) {
          fprintf(stdout, "......Interpolating point %d at (%f %f %f).\n", count, *xi, *yi, *zi);
        } else {
          fprintf(stdout, "......Interpolating point %d at (%f %f).\n", count, *xi, *yi);
        }
      }
    }

    for (int i = 0; i < dim; i++) {
      switch (directions[i]) {
        case 1:
          x[i] = *xi;
          break;
        case 2:
          x[i] = *yi;
          break;
        case 3:
          x[i] = *zi;
          break;
      }
    }

    float distance_to_point[num_support_points];
    int closest_points[num_support_points];
    if (reinitiate) {

      // The bounding box
      for (int i = 0; i < dim; i++) {
        minmaxxy[0][i] = data[0][i];
        minmaxxy[1][i] = data[0][i];
      }
      for (int i = 0; i < num_data; i++) {
        for (int j = 0; j < dim; j++) {
          if (data[i][j] < minmaxxy[0][j]) minmaxxy[0][j] = data[i][j];
          if (data[i][j] > minmaxxy[1][j]) minmaxxy[1][j] = data[i][j];
        }
      }
      max_distance = 0.0f;
      for (int i = 0; i < dim; i++) {
        max_distance = max_distance + powf((minmaxxy[1][i] - minmaxxy[0][i]),2.0f);
      }
      max_distance = 2.0f * sqrtf(max_distance);
    }

    // Get the supporting points for the interpolation
    for (int i = 0; i < num_support_points; i++) {
      distance_to_point[i] = max_distance;
    }
    memset(closest_points,-1,sizeof(closest_points));

    for (int i = 0; i < num_data; i++) {
      // Get radius to input data point
      float radius = 0.0f;
      for (int j = 0; j < dim; j++) {
        radius = radius + powf((x[j] - data[i][j]),2.0f);
      }
      if (radius > 1.0e-06) {
        radius = sqrtf(radius);
      } else {
        radius = 0.0f;
      }
      // Check whether one and if so which one
      // of the current supporting points has a longer distance
      float actual_difference = distance_to_point[num_support_points-1];
      int actual_point = 0;
      bool is_smaller_than_any = false;
      for (int j = 0; j < num_support_points; j++) {
        float difference = distance_to_point[j] -  radius;
        if (difference > 0.0f) {
          is_smaller_than_any = true;
          if (difference < actual_difference) {
            actual_point = j;
            actual_difference = difference;
          }
        }
      }
      // If so swap and reorder
      if (is_smaller_than_any) {
        for (int j = num_support_points-2; j >= actual_point; j--) {
          distance_to_point[j+1] = distance_to_point[j];
          closest_points[j+1] = closest_points[j];
        }
        distance_to_point[actual_point] = radius;
        closest_points[actual_point] = i;
      }
    }

    // Do we have a bull's eye
    if (distance_to_point[0] < 1.0e-12) {
      interpolated_value = data[closest_points[0]][dim];
    } else {
      // Interpolate
      float weight_sum = 0.0f;
      interpolated_value = 0.0f;
      int used_supporting_point = 0;
      for (int i = 0; i < num_support_points; i++) {
        if (closest_points[i] >= 0) {
          used_supporting_point = used_supporting_point + 1;
          float weight = powf(distance_to_point[i],-exponent);
          interpolated_value = interpolated_value + weight * data[closest_points[i]][dim];
          weight_sum = weight_sum + weight;
        }
      }
      if (used_supporting_point < num_support_points) {
        if (dim > 2) {
          fprintf(stdout, "Number of supporting points used for point (%f %f %f) = %d smaller than requested %d.\n",
                *xi, *yi, *zi, used_supporting_point, num_support_points);
        } else {
          fprintf(stdout, "Number of supporting points used for point (%f %f) = %d smaller than requested %d.\n",
                *xi, *yi, used_supporting_point, num_support_points);
        }
      }
      if (used_supporting_point == 0) {
        if (dim > 2) {
          fprintf(stderr, "No supporting point for point (%f %f %f) found.\n", *xi, *yi, *zi);
        } else {
          fprintf(stderr, "No supporting point for point (%f %f) found.\n", *xi, *yi);
        }
        exit(EXIT_FAILURE);
      }
      interpolated_value = interpolated_value / weight_sum;
    }

    return interpolated_value;
  }

float cpu_radially_interpolate_structured(velo_grid *velo_grid, float *xi, float *yi,
            int idx, const int NY, const int NX, const int num_support_points, const float exponent) {

    float interpolated_value;

    // Four supporting points required
    float distance_to_point[4];

    for (int i = 0; i < num_support_points; i++) {
      float radius = 0.0f;
      float ux = velo_grid->x[((NY*NX)*i)+idx];
      float uy = velo_grid->y[((NY*NX)*i)+idx];
      radius = radius + (*xi-ux)*(*xi-ux);
      radius = radius + (*yi-uy)*(*yi-uy);
      distance_to_point[i] = radius;
    }

    // Interpolate
    float weight_sum = 0.0f;
    interpolated_value = 0.0f;
    for (int i = 0; i < num_support_points; i++) {
      float weight = powf(distance_to_point[i],-exponent);
      interpolated_value = interpolated_value + weight * velo_grid->val[((NY*NX)*i)+idx];
      weight_sum = weight_sum + weight;
    }

    interpolated_value = interpolated_value / weight_sum;
    return interpolated_value;
}

#ifdef __NVCC__

__device__ float bounding_box(velo_grid *grid, const int NY, const int NX, const int dim) {

    float minmaxxy[2][3];
    float max_distance = 0.0f;

    minmaxxy[0][0] = grid->x[0];
    minmaxxy[1][0] = grid->x[0];

    minmaxxy[0][1] = grid->y[0];
    minmaxxy[1][1] = grid->y[0];

    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        if (grid->x[(y*NX)+x] < minmaxxy[0][0])
              minmaxxy[0][0] = grid->x[(y*NX)+x];
        if (grid->y[(y*NX)+x] < minmaxxy[0][1])
              minmaxxy[0][1] = grid->y[(y*NX)+x];

        if (grid->x[(y*NX)+x] > minmaxxy[1][0])
              minmaxxy[1][0] = grid->x[(y*NX)+x];
        if (grid->y[(y*NX)+x] > minmaxxy[1][1])
              minmaxxy[1][1] = grid->y[(y*NX)+x];
      }
    }

    for (int i = 0; i < dim; i++) {
        max_distance = max_distance + powf((minmaxxy[1][i] - minmaxxy[0][i]),2.0f);
    }
    max_distance = 2.0f * sqrtf(max_distance);
    return max_distance;
}

__device__ void check_point(float *distance_to_point_u, float *distance_to_point_v, float *closest_points_u,
  float *closest_points_v, int *closest_points_idx_u, int *closest_points_idx_v,
  float *radius_u, float *radius_v, float *u_val, float *v_val,
  int const NY_STAG, int const NX_STAG, int const NY, const int NX, const int z, const int *i,
  const int u, const int v, const int num_support_points, const int step) {

    // Check whether one and if so which one
    // of the current supporting points has a longer distance
    float actual_difference_u = distance_to_point_u[num_support_points-1];
    float actual_difference_v = distance_to_point_v[num_support_points-1];
    int actual_point_u = 0;
    int actual_point_v = 0;
    bool is_smaller_than_any_u = false;
    bool is_smaller_than_any_v = false;
    float r_u, r_v;
    for (int j = 0; j < num_support_points; j++) {
      if (step == 1) {
        r_u = radius_u[*i];
        r_v = radius_v[*i];
      } else {
        r_u = *radius_u;
        r_v = *radius_v;
      }
      float difference_u = distance_to_point_u[j] - r_u;
      float difference_v = distance_to_point_v[j] - r_v;
      if (difference_u > 0.0f) {
        is_smaller_than_any_u = true;
        if (difference_u < actual_difference_u) {
          actual_point_u = j;
          actual_difference_u = difference_u;
        }
      }
      if (difference_v > 0.0f) {
        is_smaller_than_any_v = true;
        if (difference_v < actual_difference_v) {
          actual_point_v = j;
          actual_difference_v = difference_v;
        }
      }
    }
    // If so swap and reorder
    if (is_smaller_than_any_u) {
      for (int j = num_support_points-2; j >= actual_point_u; j--) {
        distance_to_point_u[j+1] = distance_to_point_u[j];
        closest_points_u[j+1] = closest_points_u[j];
      }
      distance_to_point_u[actual_point_u] = r_u;
      if (step == 1) {
        closest_points_u[actual_point_u] = u_val[*i];
      } else {
        closest_points_u[actual_point_u] = u_val[(z*(NY*NX_STAG))+u];
      }
      closest_points_idx_u[actual_point_u] = (z*(NY*NX_STAG))+u;
    }

    if (is_smaller_than_any_v) {
      for (int j = num_support_points-2; j >= actual_point_v; j--) {
        distance_to_point_v[j+1] = distance_to_point_v[j];
        closest_points_v[j+1] = closest_points_v[j];
      }
      distance_to_point_v[actual_point_v] = r_v;
      if (step == 1) {
        closest_points_v[actual_point_v] = v_val[*i];
      } else {
        closest_points_v[actual_point_v] = v_val[(z*(NY_STAG*NX))+v];
      }
      closest_points_idx_v[actual_point_v] = (z*(NY_STAG*NX))+v;
    }
}

__global__ void gpu_radially_interpolate_unstructured(velo_grid *u_grid, velo_grid *v_grid, mass_grid *m_grid,
   const int NY_STAG, const int NX_STAG, const int NY, const int NX, const int z, const int dim,
   const int num_support_points, const float exponent) {

    float interpolated_value_u;
    float interpolated_value_v;
    float max_distance_u = 0.0f;
    float max_distance_v = 0.0f;

    // We assume a maximum of four supporting points
    float distance_to_point_u[4];
    float distance_to_point_v[4];
    float closest_points_u[4];
    float closest_points_v[4];
    int closest_points_idx_u[4];
    int closest_points_idx_v[4];

    int IDX = blockIdx.x * blockDim.x + threadIdx.x;

    if (IDX >= (NY*NX)) return;

    max_distance_u = bounding_box(u_grid, NY, NX_STAG, dim);
    max_distance_v = bounding_box(v_grid, NY_STAG, NX, dim);

    // Get the supporting points for the interpolation
    for (int i = 0; i < num_support_points; i++) {
      distance_to_point_u[i] = max_distance_u;
      distance_to_point_v[i] = max_distance_v;
    }
    memset(closest_points_u, 0.0f, sizeof(closest_points_u));
    memset(closest_points_v, 0.0f, sizeof(closest_points_v));
    memset(closest_points_idx_u, -1, sizeof(closest_points_idx_u));
    memset(closest_points_idx_v, -1, sizeof(closest_points_idx_v));

    int u_left = (NY*NX_STAG) % DEF_UNROLL_SIZE;
    int v_left = (NY_STAG*NX) % DEF_UNROLL_SIZE;

    float radius_u[DEF_UNROLL_SIZE];
    float radius_v[DEF_UNROLL_SIZE];

    __shared__ float ux[DEF_UNROLL_SIZE];
    __shared__ float uy[DEF_UNROLL_SIZE];
    __shared__ float vx[DEF_UNROLL_SIZE];
    __shared__ float vy[DEF_UNROLL_SIZE];
    __shared__ float u_val[DEF_UNROLL_SIZE];
    __shared__ float v_val[DEF_UNROLL_SIZE];

    for (int u=0, v=0; u<NY*NX_STAG && v<NY_STAG*NX; u+=DEF_UNROLL_SIZE, v+=DEF_UNROLL_SIZE) {

      ux[threadIdx.x] = u_grid->x[u+threadIdx.x];
      uy[threadIdx.x] = u_grid->y[u+threadIdx.x];

      vx[threadIdx.x] = v_grid->x[v+threadIdx.x];
      vy[threadIdx.x] = v_grid->y[v+threadIdx.x];

      u_val[threadIdx.x] = u_grid->val[(z*(NY*NX_STAG))+(u+threadIdx.x)];
      v_val[threadIdx.x] = v_grid->val[(z*(NY_STAG*NX))+(v+threadIdx.x)];

      __syncthreads();

      for (int i = 0; i < DEF_UNROLL_SIZE; i++) {

        int uu = u+i;
        int vv = v+i;

        radius_u[i] = 0.0f;
        radius_u[i] = radius_u[i] + (m_grid->x[IDX]-ux[i])*(m_grid->x[IDX]-ux[i]);
        radius_u[i] = radius_u[i] + (m_grid->y[IDX]-uy[i])*(m_grid->y[IDX]-uy[i]);

        radius_v[i] = 0.0f;
        radius_v[i] = radius_v[i] + (m_grid->x[IDX]-vx[i])*(m_grid->x[IDX]-vx[i]);
        radius_v[i] = radius_v[i] + (m_grid->y[IDX]-vy[i])*(m_grid->y[IDX]-vy[i]);

        if (radius_u[i] > 1.0e-06) {
          radius_u[i] = sqrtf(radius_u[i]);
        } else {
          radius_u[i] = 0.0f;
        }
        if (radius_v[i] > 1.0e-06) {
          radius_v[i] = sqrtf(radius_v[i]);
        } else {
          radius_v[i] = 0.0f;
        }

        check_point(distance_to_point_u, distance_to_point_v, closest_points_u,
                    closest_points_v, closest_points_idx_u, closest_points_idx_v,
                    radius_u, radius_v, u_val, v_val,
                    NY_STAG, NX_STAG, NY, NX, z, &i,
                    uu, vv, num_support_points, 1);
      } // End of 16 elements block
    }

    for (int u=(NY*NX_STAG)-u_left, v=(NY_STAG*NX)-v_left; u<(NY*NX_STAG) && v<(NY_STAG*NX); u++, v++) {

      float radius_u = 0.0f;
      float radius_v = 0.0f;
      radius_u = radius_u + (m_grid->x[IDX]-u_grid->x[u])*(m_grid->x[IDX]-u_grid->x[u]);
      radius_u = radius_u + (m_grid->y[IDX]-u_grid->y[u])*(m_grid->y[IDX]-u_grid->y[u]);

      radius_v = radius_v + (m_grid->x[IDX]-v_grid->x[v])*(m_grid->x[IDX]-v_grid->x[v]);
      radius_v = radius_v + (m_grid->y[IDX]-v_grid->y[v])*(m_grid->y[IDX]-v_grid->y[v]);

      if (radius_u > 1.0e-06) {
           radius_u = sqrtf(radius_u);
      } else {
           radius_u = 0.0f;
      }
      if (radius_v > 1.0e-06) {
           radius_v = sqrtf(radius_v);
      } else {
           radius_v = 0.0f;
      }

      check_point(distance_to_point_u, distance_to_point_v, closest_points_u,
                  closest_points_v, closest_points_idx_u, closest_points_idx_v,
                  &radius_u, &radius_v, u_grid->val, v_grid->val,
                  NY_STAG, NX_STAG, NY, NX, z, NULL,
                  u, v, num_support_points, 2);
    }

    // Do we have a bull's eye
    if (distance_to_point_u[0] < 1.0e-12 || distance_to_point_v[0] < 1.0e-12) {
      if (distance_to_point_u[0] < 1.0e-12) interpolated_value_u = closest_points_u[0];
      if (distance_to_point_v[0] < 1.0e-12) interpolated_value_v = closest_points_v[0];
    } else {
      // Interpolate
      float weight_sum_u = 0.0f;
      float weight_sum_v = 0.0f;
      interpolated_value_u = 0.0f;
      interpolated_value_v = 0.0f;
      int used_supporting_point_u = 0;
      int used_supporting_point_v = 0;
      for (int i = 0; i < num_support_points; i++) {
        if (closest_points_idx_u[i] >= 0) {
          used_supporting_point_u = used_supporting_point_u + 1;
          float weight = powf(distance_to_point_u[i],-exponent);
          interpolated_value_u = interpolated_value_u + weight * closest_points_u[i];
          weight_sum_u = weight_sum_u + weight;
        }
      }
      for (int i = 0; i < num_support_points; i++) {
        if (closest_points_idx_v[i] >= 0) {
          used_supporting_point_v = used_supporting_point_v + 1;
          float weight = powf(distance_to_point_v[i],-exponent);
          interpolated_value_v = interpolated_value_v + weight * closest_points_v[i];
          weight_sum_v = weight_sum_v + weight;
        }
      }

      if (used_supporting_point_u == 0 || used_supporting_point_v == 0) {
        printf("Failure in interpolation due to lack of supporting points.\n");
      }

      interpolated_value_u = interpolated_value_u / weight_sum_u;
      interpolated_value_v = interpolated_value_v / weight_sum_v;
    }

    m_grid->u[IDX] = interpolated_value_u;
    m_grid->v[IDX] = interpolated_value_v;
}

__global__ void gpu_radially_interpolate_structured(velo_grid *u_grid, velo_grid *v_grid, mass_grid *m_grid,
   const int NY, const int NX, const int num_support_points, const float exponent) {

    float interpolated_value_u;
    float interpolated_value_v;

    // Four supporting points required
    float distance_to_point_u[4];
    float distance_to_point_v[4];

    int IDX = blockIdx.x * blockDim.x + threadIdx.x;

    if (IDX >= (NY*NX)) return;

    for (int i = 0; i < num_support_points; i++) {
      float radius_u = 0.0f;
      float ux = u_grid->x[((NY*NX)*i)+IDX];
      float uy = u_grid->y[((NY*NX)*i)+IDX];
      radius_u = radius_u + (m_grid->x[IDX]-ux)*(m_grid->x[IDX]-ux);
      radius_u = radius_u + (m_grid->y[IDX]-uy)*(m_grid->y[IDX]-uy);

      float radius_v = 0.0f;
      float vx = v_grid->x[((NY*NX)*i)+IDX];
      float vy = v_grid->y[((NY*NX)*i)+IDX];
      radius_v = radius_v + (m_grid->x[IDX]-vx)*(m_grid->x[IDX]-vx);
      radius_v = radius_v + (m_grid->y[IDX]-vy)*(m_grid->y[IDX]-vy);

      distance_to_point_u[i] = radius_u;
      distance_to_point_v[i] = radius_v;
    }

    // Interpolate
    float weight_sum_u = 0.0f;
    float weight_sum_v = 0.0f;
    interpolated_value_u = 0.0f;
    interpolated_value_v = 0.0f;
    for (int i = 0; i < num_support_points; i++) {
      float weight = powf(distance_to_point_u[i],-exponent);
      interpolated_value_u = interpolated_value_u + weight * u_grid->val[((NY*NX)*i)+IDX];
      weight_sum_u = weight_sum_u + weight;
    }
    for (int i = 0; i < num_support_points; i++) {
      float weight = powf(distance_to_point_v[i],-exponent);
      interpolated_value_v = interpolated_value_v + weight * v_grid->val[((NY*NX)*i)+IDX];
      weight_sum_v = weight_sum_v + weight;
    }

    interpolated_value_u = interpolated_value_u / weight_sum_u;
    interpolated_value_v = interpolated_value_v / weight_sum_v;

    m_grid->u[IDX] = interpolated_value_u;
    m_grid->v[IDX] = interpolated_value_v;
}
#endif
