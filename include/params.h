// The working directory
#define WORKDIR "/home/seddik/Documents/workdir/WRF_Jebi"

// The date of the files we are processing
#define DATE "2018"

// The number of supporting points for the interpolation
#define NUM_SUPPORTING_POINTS 4

#define UNROLL_SIZE 256
#define BLOCK_SIZE 128

enum {
  STRUCTURED=1,
  UNSTRUCTURED
};

enum {
  NORMALIZATION=1,
  NORMALIZATION_CENTERED,
  STANDARDIZATION
};

#define GRID_TYPE STRUCTURED
#define FEATURE_SCALING STANDARDIZATION

// Pointer to the feature scaling routine
typedef float (*feature_scaling_pt)(float arg1, float *arg2, uint arg3, bool *arg4);
