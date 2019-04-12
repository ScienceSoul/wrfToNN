#include "memory.h"

tensor *allocate_tensor(uint *shape, uint rank) {

  tensor * t = (tensor *)malloc(sizeof(tensor));
  if (t == NULL) {
    perror("tensor allocation problem");
    return NULL;
  }

  uint size = 1;
  for (int i = 0; i < rank; i++) {
    size = size * shape[i];
  }

  t->val = (float *)malloc(size*sizeof(float));
  if (t->val == NULL) {
    perror("tensor values allocation problem");
    return NULL;
  }

  memcpy(t->shape, shape, rank*sizeof(uint));
  t->rank = rank;

  return t;
}

void deallocate_tensor(tensor *t) {
 free(t->val);
 free(t);
}

map *allocate_maps(int num_variables) {

  map *m = (map *)malloc(num_variables * sizeof(map));
  for (int i = 0; i < num_variables; i++) {
    m[i].name = NULL;
    m[i].out_name = NULL;
    m[i].variable = NULL;
    m[i].mass_variable = NULL;
    m[i].longi = NULL;
    m[i].lat = NULL;
    m[i].mass_longi = NULL;
    m[i].mass_lat = NULL;
    m[i].active = false;
  }
  return m;
}

float **allocate_2d(uint dim1, uint dim2) {

    float **t = NULL;

    t = (float **)malloc(dim1 * sizeof(float*));
    if (t == NULL) {
      perror("allocate_2d: allocation error in first dimension");
      return NULL;
    }

    for (int i = 0; i < dim1; i++) {
      if ((t[i] = (float *)malloc(dim2 * sizeof(float))) == NULL) {
        perror("allocate_2d: allocation error in second dimension");
      }
    }

    return t;
}

float ***allocate_3d(uint dim1, uint dim2, uint dim3) {

  float ***t = NULL;

  t = (float ***)malloc(dim1 * sizeof(float **));
  if (t == NULL) {
    perror("allocate_3d: allocation error in first dimension");
    return NULL;
  }

  for (int i = 0; i < dim1; i++) {
    if ((t[i] = (float **)malloc(dim2 * sizeof(float *))) == NULL) {
      perror("allocate_3d: allocation error in second dimension");
      return NULL;
    }
    for (int j = 0; j < dim2; j++) {
      if((t[i][j] = (float *)malloc(dim3 * sizeof(float))) == NULL) {
        perror("allocate_3d: allocation error in third dimension");
        return NULL;
      }
    }
  }

  return t;
}

float ****allocate_4d(uint dim1, uint dim2, uint dim3, uint dim4) {

  float ****t = NULL;

  t = (float ****)malloc(dim1 * sizeof(float ***));
  if (t == NULL) {
    perror("allocate_4d: allocation error in first dimension");
    return NULL;
  }

  for (int i = 0; i < dim1; i++) {
    if ((t[i] = (float ***)malloc(dim2 * sizeof(float **))) == NULL) {
      perror("allocate_4d: allocation error in second dimension");
      return NULL;
    }
    for (int j = 0; j < dim2; j++) {
      if ((t[i][j] = (float **)malloc(dim3 * sizeof(float *))) == NULL) {
        perror("allocate_4d: allocation error in third dimension");
        return NULL;
      }
      for (int k = 0; k < dim3; k++) {
        if ((t[i][j][k] = (float *)malloc(dim4 * sizeof(float))) == NULL) {
          perror("allocate_4d: allocation error in fourth dimension");
          return NULL;
        }
      }
    }
  }

  return t;
}

void deallocate_2d(float **t, uint dim1) {

  for (int i = 0; i < dim1; i++) {
    free(t[i]);
  }
  free(t);
}

void deallocate_3d(float ***t, uint dim1, uint dim2) {

  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      free(t[i][j]);
    }
    free(t[i]);
  }

  free(t);
}

void deallocate_4d(float ****t, uint dim1, uint dim2, uint dim3) {

  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      for (int k = 0; k < dim3; k++) {
        free(t[i][j][k]);
      }
      free(t[i][j]);
    }
    free(t[i]);
  }

  free(t);
}
