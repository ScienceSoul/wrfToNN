# ifndef memory_h
#define memory_h

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MAX_TENSOR_RANK 4
#define MAX_STRING_LENGTH 1024
#define MAX_NUMBER_FILES 500
#define MAX_NUMBER_VARIABLES 200

typedef struct tensor {
    uint shape[MAX_TENSOR_RANK];
    uint rank;
    float *val;
} tensor;

typedef struct map {
  const char *name;
  const char *out_name;
  tensor *variable;
  tensor *mass_variable;
  tensor *longi;
  tensor *lat;
  tensor *mass_longi;
  tensor *mass_lat;
  bool active;
} map;

tensor *allocate_tensor(uint *shape, uint rank);
void deallocate_tensor(tensor *t);
map *allocate_maps(int num_variables);

float **allocate_2d(uint dim1, uint dim2);
float ***allocate_3d(uint dim1, uint dim2, uint dim3);
float ****allocate_4d(uint dim1, uint dim2, uint dim3, uint dim4);

void deallocate_2d(float **t, uint dim1);
void deallocate_3d(float ***t, uint dim1, uint dim2);
void deallocate_4d(float ****t, uint dim1, uint dim2, uint dim3);

#endif
