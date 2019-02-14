#include <math.h>
#include "feature_scaling.h"

float maxv(float *vec, uint length) {
  float max = -HUGE_VALF;
  for (int i = 0; i < length; i++) {
    if (vec[i] > max) {
      max = vec[i];
    }
  }
  return max;
}

float minv(float *vec, uint length) {
  float min = HUGE_VALF;
  for (int i = 0; i < length; i++) {
    if (vec[i] < min) {
      min = vec[i];
    }
  }
  return min;
}

float meanv(float *vec, uint length) {
  float sum = 0.0f;
  for (int i = 0; i < length; i++) {
    sum = sum + vec[i];
  }
  return sum / length;
}

float standard_dev(float mean, float *vec, uint length) {
  float s = 0.0f;
  for (int i = 0; i < length; i++) {
    s = s + powf((vec[i] - mean), 2.0f);
  }
  return sqrtf(s/float(length-1));

}

// Normalize in range [0,1]
float normalize(float current, float *vec, uint length, bool *new_call) {

  static bool init = false;
  static float max = 0, min = 0;

  if (*new_call) {
    init = true;
  }
  if (init) {
    max = maxv(vec, length);
    min = minv(vec, length);
    init = false;
    *new_call = false;
  }

  return (current - min) / (max - min);
}

// Nornalize in range [-1,1] centered at 0
float normalize_center(float current, float *vec, uint length, bool *new_call) {

  static bool init = false;
  static float max = 0, min = 0;

  if (*new_call) {
    init = true;
  }
  if (init) {
    max = maxv(vec, length);
    min = minv(vec, length);
    init = false;
    *new_call = false;
  }

  return ( current - ((max + min)/2) ) / ( (max - min)/2.0f );
}

// Standardize to 0 mean and 1 variance
float standardize(float current, float *vec, uint length, bool *new_call) {

  static bool init = false;
  static float mean = 0, std = 0;

  if (*new_call) {
    init = true;
  }
  if (init) {
    mean = meanv(vec, length);
    std = standard_dev(mean, vec, length);
    init = false;
    *new_call = false;
  }

  return (current - mean) / std;
}
