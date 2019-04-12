#ifndef params_h
#define params_h

// The working directory
#define WORKDIR "/home/seddik/Documents/workdir/WRF_Jebi"

// The date of the files we are processing
#define DATE "2018"

// Stack size
#define STACK_SIZE 16 // MB

// The number of supporting points for the interpolation
#define NUM_SUPPORTING_POINTS 4

#define DEF_BLOCK_SIZE 128

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

// ----------------------------------
// The global dimensions
// ----------------------------------
#define DEF_NX 660;
#define DEF_NX_STAG 661;

#define DEF_NY 710;
#define DEF_NY_STAG 711;

#define DEF_NZ 50;
#define DEF_NZ_STAG 51;
#define DEF_NZ_SOIL_STAG 4;

#define DEF_NT 1;
// ----------------------------------

// ------------------------------------------------------
// The variables definition
// ------------------------------------------------------
typedef enum variables_code {
    //-----------------------------------
    // 1D variables
    //-----------------------------------
    ZNU=0,     // Eta values on half (mass) levels
    ZNW,       // Eta values on full (w) levels

    //-----------------------------------
    // 2D variables
    //-----------------------------------
    XLAT,      // Latitude
    XLAT_U,    // x-wind latitude
    XLAT_V,    // y-wind latitude
    XLONG,     // Longitute
    XLONG_U,   // x-wind longitude
    XLONG_V,   // y-wind longitude
    SST,       // Sea surface temperature
    OLR,       // TOA outgoing long wave

    //-----------------------------------
    // 3D variables
    // ----------------------------------
    CLDFRA,    // Cloud fraction
    P,         // Perturbation pressure
    PB,        // Base state pressure
    PH,        // Perturbation geopotential
    PHB,       // Base-state geopotential
    P_HYD,     // hydrostatic presure
    QCLOUD,    // Cloud water mixing fraction
    QGRAUP,    // Graupel mixing ratio
    QICE,      // Ice mixing ratio
    QNGRAUPEL, // Graupel number concentration
    QNICE,     // Ice number concentration
    QNRAIN,    // Rain number concentration
    QNSNOW,    // Snow number concentration
    QRAIN,     // Rain water mixing ratio
    QSNOW,     // Snow mixing ratio
    QVAPOR,    // Water vapor mixing ratio
    SH2O,      // Soil liquid water
    SMCREL,    // Relative soil moisture
    SMOIS,     // Soil moisture
    T,         // Perturbation potential temperature (theta-t0)
    TSLB,      // Soil temperature
    U,         // x-wind component
    V,         // y-wind component
    W          // z-wind component
} variables_code;

#define DEF_NUM_VARIABLES 34;

const char *active_flags[] = {"ZNU:1",
                              "ZNW:1",
                              "XLAT:0",
                              "XLAT_U:0",
                              "XLAT_V:0",
                              "XLONG:0",
                              "XLONG_U:0",
                              "XLONG_V:0",
                              "SST:0",
                              "OLR:0",
                              "CLDFRA:0",
                              "P:0",
                              "PB:0",
                              "PH:0",
                              "PHB:0",
                              "P_HYD:1",
                              "QCLOUD:0",
                              "QICE:0",
                              "QNGRAUPEL:0",
                              "QNICE:0",
                              "QNRAIN:0",
                              "QNSNOW:0",
                              "QRAIN:0",
                              "QSNOW:0",
                              "QVAPOR:1",
                              "SH20:0",
                              "SMCREL:0",
                              "SMOIS:0",
                              "T:1",
                              "TSLB:0",
                              "U:1",
                              "V:1",
                              "W:0"};

#endif
