#ifndef feature_scaling_h
#define feature_scaling_h

#include <xmmintrin.h>

float x86_sse_vsum(float *vec, uint length);
void x86_sse_min_max(float *buf, unsigned nframes, float *min, float *max);

float maxv(float *vec, uint length);
float minv(float *vec, uint length);
float meanv(float *vec, uint length);
float standard_dev(float mean, float *vec, uint length);

float normalize(float current, float *vec, uint length, bool *new_call);
float normalize_centered(float current, float *vec, uint length, bool *new_call);
float standardize(float current, float *vec, uint length, bool *new_call);

#endif
