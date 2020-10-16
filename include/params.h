#ifndef params_h
#define params_h

// The working directory
#define WORKDIR "/home/seddik/Documents/workdir/weather_control/ncep_gfs/simul"

// The date of the files we are processing
#define DATE "2020"

// The domain identifier used in the WRF outputs
#define DOMAIN "d01"

// Define tne name of the grid spacing attributes
#define DX_NAME "DX"
#define DY_NAME "DY"

// Define tne name of the centereed latitude and longitude attributes
#define CEN_LAT_NAME  "CEN_LAT"
#define CEN_LONG_NAME "CEN_LON"

// Stack size
#define STACK_SIZE 16 // MB

// The number of supporting points for the interpolation
// in an unstructured grid
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

// -----------------------------------------------------
// The global dimensions
// -----------------------------------------------------
#define DEF_NX 660;
#define DEF_NX_STAG 661;

#define DEF_NY 710;
#define DEF_NY_STAG 711;

#define DEF_NZ 50;
#define DEF_NZ_STAG 51;
#define DEF_NZ_SOIL_STAG 4;

#define DEF_NT 1;
// -----------------------------------------------------

// -----------------------------------------------------
float VORTICITY_SCALING = 1.0e3f;

// For visualization purpose
bool  VORTICITY_LIMITER = false;
float VORTICITY_LIMITER_LOW = 1.0f;
// -----------------------------------------------------

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
    MU,        // Perturbation dry air mass in column
    MUB,       // Base state dry air mass in column
    DRY_MASS,  // The full dry air mass in column = MU + MUB
    OLR,       // TOA outgoing long wave
    //-----------------------------------

    // 3D variables
    // ----------------------------------
    CLDFRA,        // Cloud fraction
    P,             // Perturbation pressure
    PB,            // Base state pressure
    PH,            // Perturbation geopotential
    PHB,           // Base-state geopotential
    P_HYD,         // hydrostatic presure
    QCLOUD,        // Cloud water mixing fraction
    QGRAUP,        // Graupel mixing ratio
    QICE,          // Ice mixing ratio
    QNGRAUPEL,     // Graupel number concentration
    QNICE,         // Ice number concentration
    QNRAIN,        // Rain number concentration
    QNSNOW,        // Snow number concentration
    QRAIN,         // Rain water mixing ratio
    QSNOW,         // Snow mixing ratio
    QVAPOR,        // Water vapor mixing ratio
    SH2O,          // Soil liquid water
    SMCREL,        // Relative soil moisture
    SMOIS,         // Soil moisture
    PERT_T,        // Perturbation potential temperature (theta-t0)
    TEMP,          // Temperature (computed from the perturbation potential temperature)
    TSLB,          // Soil temperature
    U,             // x-wind component
    V,             // y-wind component
    W,             // z-wind component
    PRESSURE,      // The full pressure = P + PB
    COR_EAST,      // Coriolis force directed due East
    COR_NORTH,     // Coriolis force directed due North
    GEOPOTENTIAL,  // The full geopotential = PH + PHB
    COR_PARAM,     // Coriolis parameter 2*\omega*sin(\phy)
    ABS_VERT_VORT, // Absolute vertical
                   // vorticity = f + reference vertical vorticity
    REL_VERT_VORT  // Relative vertical vorticity dv/dx - du/dy
} variables_code;

#define DEF_NUM_VARIABLES 45;

const char *active_flags[] = {"ZNU:1",
                              "ZNW:1",
                              "XLAT:0",
                              "XLAT_U:0",
                              "XLAT_V:0",
                              "XLONG:0",
                              "XLONG_U:0",
                              "XLONG_V:0",
                              "SST:0",
                              "MU:0",
                              "MUB:0",
                              "DRY_MASS:1",
                              "OLR:0",
                              "CLDFRA:0",
                              "P:0",
                              "PB:0",
                              "PH:0",
                              "PHB:0",
                              "P_HYD:1",
                              "QCLOUD:0",
                              "QGRAUP:0",
                              "QICE:0",
                              "QNGRAUPEL:0",
                              "QNICE:0",
                              "QNRAIN:0",
                              "QNSNOW:0",
                              "QRAIN:1",
                              "QSNOW:0",
                              "QVAPOR:1",
                              "SH20:0",
                              "SMCREL:0",
                              "SMOIS:0",
                              "T:1",
                              "TEMP:1",
                              "TSLB:0",
                              "U:1",
                              "V:1",
                              "W:1",
                              "PRESSURE:1",
                              "COR_EAST:0",
                              "COR_NORTH:0",
                              "GEOPOTENTIAL:0",
                              "COR_PARAM:0",
                              "ABS_VERT_VORT:0",
                              "REL_VERT_VORT:0"};

#endif
