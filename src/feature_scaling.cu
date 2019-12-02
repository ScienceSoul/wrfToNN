#include <math.h>
#include <stdio.h>
#include "feature_scaling.h"
#include "gpu.h"

float x86_sse_vsum(float *vec, uint length) {

  __m128 vv, vs;
  float s = 0.0f;
  int i;

  vs = _mm_setzero_ps();
  for (i = 0; i < length-4+1; i+=4) {
    vv = _mm_loadu_ps(&vec[i]);
    vs = _mm_add_ps(vv, vs);
  }
  s = vs[0] + vs[1] + vs[2] + vs[3];
  for (; i < length; i++) {
    s += vec[i];
  }
  return s;
}

void x86_sse_min_max(float *vec, unsigned length, float *min, float *max) {

  __m128 current_max, current_min, work;

  // Load max and min values into all four slots of the XMM registersmin
  current_min = _mm_set1_ps(*min);
  current_max = _mm_set1_ps(*max);

  // Work input until "vec" reaches 16 byte alignment
  while ( ((unsigned long)vec) % 16 != 0 && length > 0) {

    // Load the next float into the work buffer
    work = _mm_set1_ps(*vec);

    current_min = _mm_min_ps(current_min, work);
    current_max = _mm_max_ps(current_max, work);

    vec++;
    length--;
  }

  // use 64 byte prefetch for quadruple quads
  while (length >= 16) {

    work = _mm_load_ps(vec);
    current_min = _mm_min_ps(current_min, work);
    current_max = _mm_max_ps(current_max, work);
    vec+=4;
    work = _mm_load_ps(vec);
    current_min = _mm_min_ps(current_min, work);
    current_max = _mm_max_ps(current_max, work);
    vec+=4;
    work = _mm_load_ps(vec);
    current_min = _mm_min_ps(current_min, work);
    current_max = _mm_max_ps(current_max, work);
    vec+=4;
    work = _mm_load_ps(vec);
    current_min = _mm_min_ps(current_min, work);
    current_max = _mm_max_ps(current_max, work);
    vec+=4;
    length-=16;
  }

  // work through aligned buffers
  while (length >= 4) {

    work = _mm_load_ps(vec);

    current_min = _mm_min_ps(current_min, work);
    current_max = _mm_max_ps(current_max, work);

    vec+=4;
    length-=4;
  }

  // work through the rest < 4 samples
  while ( length > 0) {

    // Load the next float into the work buffer
    work = _mm_set1_ps(*vec);

    current_min = _mm_min_ps(current_min, work);
    current_max = _mm_max_ps(current_max, work);

    vec++;
    length--;
  }

  // Find min & max value in current_max through shuffle tricks

  work = current_min;
  work = _mm_shuffle_ps(work, work, _MM_SHUFFLE(2, 3, 0, 1));
  work = _mm_min_ps (work, current_min);
  current_min = work;
  work = _mm_shuffle_ps(work, work, _MM_SHUFFLE(1, 0, 3, 2));
  work = _mm_min_ps (work, current_min);

  _mm_store_ss(min, work);

  work = current_max;
  work = _mm_shuffle_ps(work, work, _MM_SHUFFLE(2, 3, 0, 1));
  work = _mm_max_ps (work, current_max);
  current_max = work;
  work = _mm_shuffle_ps(work, work, _MM_SHUFFLE(1, 0, 3, 2));
  work = _mm_max_ps (work, current_max);

  _mm_store_ss(max, work);
}

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
  float s = 0.0f;
  for (int i = 0; i < length; i++) {
    s = s + vec[i];
  }
  return s / length;
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
  static float max, min;

  if (*new_call) {
    init = true;
  }
  if (init) {
    double rt = cpu_second();
  #ifdef __SSE__
    min = HUGE_VALF;
    max = -HUGE_VALF;
    x86_sse_min_max(vec, length, &min, &max);
  #else
    max = maxv(vec, length);
    min = minv(vec, length);
  #endif
    printf("---- max/min <%f,%f> on %d nodes (%f sec).\n", max, min, length, cpu_second() - rt);
    init = false;
    *new_call = false;
  }

  return (current - min) / (max - min);
}

// Nornalize in range [-1,1] centered at 0
float normalize_centered(float current, float *vec, uint length, bool *new_call) {

  static bool init = false;
  static float max, min;

  if (*new_call) {
    init = true;
  }
  if (init) {
    double rt = cpu_second();
  #ifdef __SSE__
    min = HUGE_VALF;
    max = -HUGE_VALF;
    x86_sse_min_max(vec, length, &min, &max);
  #else
    max = maxv(vec, length);
    min = minv(vec, length);
  #endif
    printf("---- max/min <%f,%f> on %d nodes (%f sec).\n", max, min, length, cpu_second() - rt);
    init = false;
    *new_call = false;
  }

  return ( current - ((max + min)/2) ) / ( (max - min)/2.0f );
}

// Standardize to 0 mean and 1 variance
float standardize(float current, float *vec, uint length, bool *new_call) {

  static bool init = false;
  static float mean, std;

  if (*new_call) {
    init = true;
  }
  if (init) {
    double rt = cpu_second();
#ifdef __SSE__
    float sum = x86_sse_vsum(vec, length);
    mean = sum / length;
#else
    mean = meanv(vec, length);
#endif
    std = standard_dev(mean, vec, length);
    printf("---- mean/std <%f,%f> on %d nodes (%f sec).\n", mean, std, length, cpu_second() - rt);
    init = false;
    *new_call = false;
  }

  return (current - mean) / std;
}
