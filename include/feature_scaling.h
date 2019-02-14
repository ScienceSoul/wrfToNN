#ifndef feature_scaling_h
#define feature_scaling_h

float maxv(float *vec, uint length);
float minv(float *vec, uint length);
float meanv(float *vec, uint length);
float standard_dev(float mean, float *vec, uint length);

float normalize(float current, float *vec, uint length, bool *new_call);
float normalize_center(float current, float *vec, uint length, bool *new_call);
float standardize(float current, float *vec, uint length, bool *new_call);

#endif
