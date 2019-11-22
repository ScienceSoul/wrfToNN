#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#include <netcdf.h>

#include "params.h"
#include "interpolate.h"
#ifdef __NVCC__
  #include "gpu.h"
#endif
#include "feature_scaling.h"
#include "finite_difference.h"

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

// Pointer to the feature scaling routine
typedef float (*feature_scaling_pt)(float arg1, float *arg2, uint arg3, bool *arg4);

// ------------------------------------------------------
uint NX           = DEF_NX;
uint NX_STAG      = DEF_NX_STAG;
uint NY           = DEF_NY;
uint NY_STAG      = DEF_NY_STAG;
uint NZ           = DEF_NZ;
uint NZ_STAG      = DEF_NZ_STAG;
uint NZ_SOIL_STAG = DEF_NZ_SOIL_STAG;
uint NT           = DEF_NT;

uint NUM_VARIABLES = DEF_NUM_VARIABLES;

uint UNROLL_SIZE = DEF_UNROLL_SIZE;
uint BLOCK_SIZE = DEF_BLOCK_SIZE;
// ------------------------------------------------------

// ------------------------------------------------------
// The tensors
// ------------------------------------------------------
tensor *znu           = NULL;
tensor *znw           = NULL;
tensor *xlat          = NULL;
tensor *xlong         = NULL;
tensor *xlat_u        = NULL;
tensor *xlong_u       = NULL;
tensor *xlat_v        = NULL;
tensor *xlong_v       = NULL;
tensor *sst           = NULL;
tensor *mu            = NULL;
tensor *mub           = NULL;
tensor *dry_mass      = NULL;
tensor *olr           = NULL;
tensor *cldfra        = NULL;
tensor *p             = NULL;
tensor *pb            = NULL;
tensor *phyd          = NULL;
tensor *qcloud        = NULL;
tensor *qgraup        = NULL;
tensor *qice          = NULL;
tensor *qngraupel     = NULL;
tensor *qnice         = NULL;
tensor *qnrain        = NULL;
tensor *qnsnow        = NULL;
tensor *qrain         = NULL;
tensor *qsnow         = NULL;
tensor *qvapor        = NULL;
tensor *t             = NULL;
tensor *ph            = NULL;
tensor *phb           = NULL;
tensor *w             = NULL;
tensor *sh2o          = NULL;
tensor *smcrel        = NULL;
tensor *smois         = NULL;
tensor *tslb          = NULL;
tensor *u             = NULL;
tensor *v             = NULL;
tensor *pressure      = NULL;
tensor *cor_east      = NULL;
tensor *cor_north     = NULL;
tensor *geopotential  = NULL;
tensor *cor_param     = NULL;
tensor *abs_vert_vort = NULL;
tensor *rel_vert_vort = NULL;
// ------------------------------------------------------

// ------------------------------------------------------
// The finite difference tags
// ------------------------------------------------------
fd_tags *domain_tags = NULL;
// ------------------------------------------------------

// ------------------------------------------------------
// Used for CUDA when available
// ------------------------------------------------------
#ifdef __NVCC__
  velo_grid *h_velo_u_grid = NULL;
  velo_grid *h_velo_v_grid = NULL;
  mass_grid *h_mass_grid   = NULL;

  velo_grid *d_velo_u_grid = NULL;
  velo_grid *d_velo_v_grid = NULL;
  mass_grid *d_mass_grid   = NULL;

  velo_grid *h_velo_w_grid = NULL;
  velo_grid *d_velo_w_grid = NULL;

  velo_grid *h_base_geopot_grid = NULL;
  velo_grid *d_base_geopot_grid = NULL;

  velo_grid *h_pert_geopot_grid = NULL;
  velo_grid *d_pert_geopot_grid = NULL;

  fd_container *h_abs_vert_vort = NULL;
  fd_container *d_abs_vert_vort = NULL;

  fd_container *h_rel_vert_vort = NULL;
  fd_container *d_rel_vert_vort = NULL;
#endif
// ------------------------------------------------------

// ----------------------------------
// The variable->tensor mappings
// ----------------------------------
map *maps = NULL;
// ----------------------------------

// ------------------------------------------------
// Globals for the centered latitude and longitude
// ------------------------------------------------
float cen_lat;
float cen_long;
// ------------------------------------------------

// ----------------------------------------------
// Globals for the grid spacing
// ----------------------------------------------
float dx;
float dy;
// ----------------------------------------------

// --------------------------------------------------
// Globals for the latitude and longitude increments
// --------------------------------------------------
float d_lat;
float d_long;
// ---------------------------------------------------

// ---------------------------------------------------
// Some globals
// ---------------------------------------------------
const float earth_radius = 6371.0f;
const float earth_angular_velocity = 7.2921e-5; // rad/s
// ---------------------------------------------------

void set_maps(map *maps, bool initial) {

  for (int i = 0; i < NUM_VARIABLES; i++) {

    switch (i) {
      case ZNU:
        if (initial) {
          maps[i].name = "ZNU";
          maps[i].out_name = "ZNU";
        } else {
          maps[i].variable = znu;
          maps[i].longi = znu; // Just for dummy
          maps[i].lat = znu;   // Just for dummy
        }
        break;
      case ZNW:
        if (initial) {
          maps[i].name = "ZNW";
          maps[i].out_name = "ZNW";
        } else {
          maps[i].variable = znw;
          maps[i].longi = znw; // Just for dummy
          maps[i].lat = znw;   // Just for dummy
        }
        break;
      case XLAT:
        if (initial) {
          maps[i].name = "XLAT";
        } else {
          maps[i].variable = xlat;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case XLAT_U:
        if (initial) {
          maps[i].name = "XLAT_U";
        } else {
          maps[i].variable = xlat_u;
          maps[i].longi = xlong_u;
          maps[i].lat = xlat_u;
        }
        break;
      case XLAT_V:
        if (initial) {
          maps[i].name = "XLAT_V";
        } else {
          maps[i].variable = xlat_v;
          maps[i].longi = xlong_v;
          maps[i].lat = xlat_v;
        }
        break;
      case XLONG:
        if (initial) {
        maps[i].name = "XLONG";
        } else {
          maps[i].variable = xlong;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case XLONG_U:
        if (initial) {
          maps[i].name = "XLONG_U";
        } else {
          maps[i].variable = xlong_u;
          maps[i].longi = xlong_u;
          maps[i].lat = xlat_u;
        }
        break;
      case XLONG_V:
        if (initial) {
          maps[i].name = "XLONG_V";
        } else {
          maps[i].variable = xlong_v;
          maps[i].longi = xlong_v;
          maps[i].lat = xlat_v;
        }
        break;
      case SST:
        if (initial) {
          maps[i].name = "SST";
          maps[i].out_name = "SST";
        } else {
          maps[i].variable = sst;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case MU:
        if (initial) {
          maps[i].name = "MU";
          maps[i].out_name = "MU";
        } else {
          maps[i].variable = mu;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case MUB:
        if (initial) {
          maps[i].name = "MUB";
          maps[i].out_name = "MUB";
        } else {
          maps[i].variable = mub;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case DRY_MASS:
        if (initial) {
          maps[i].name = "DRY_MASS";
          maps[i].out_name = "DRY_MASS";
        } else {
          maps[i].variable = dry_mass;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case OLR:
        if (initial) {
          maps[i].name = "OLR";
          maps[i].out_name = "OLR";
        } else {
          maps[i].variable = olr;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case CLDFRA:
        if (initial) {
          maps[i].name = "CLDFRA";
          maps[i].out_name = "CLDFRA";
        } else {
          maps[i].variable = cldfra;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case P:
        if (initial) {
          maps[i].name = "P";
          maps[i].out_name = "PERT_P";
        } else {
          maps[i].variable = p;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case PB:
        if (initial) {
          maps[i].name = "PB";
          maps[i].out_name = "PB";
        } else {
          maps[i].variable = pb;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case PH:
        if (initial) {
          maps[i].name = "PH";
          maps[i].out_name = "PH";
        } else {
          maps[i].variable = ph;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case PHB:
        if (initial) {
          maps[i].name = "PHB";
          maps[i].out_name = "PHB";
        } else {
          maps[i].variable = phb;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case P_HYD:
        if (initial) {
          maps[i].name = "P_HYD";
          maps[i].out_name = "P_HYD";
        } else {
          maps[i].variable = phyd;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case QCLOUD:
        if (initial) {
          maps[i].name = "QCLOUD";
          maps[i].out_name = "QCLOUD";
        } else {
          maps[i].variable = qcloud;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case QGRAUP:
        if (initial) {
          maps[i].name = "QGRAUP";
          maps[i].out_name = "QGRAUP";
        } else {
          maps[i].variable = qgraup;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case QICE:
        if (initial) {
          maps[i].name = "QICE";
          maps[i].out_name = "QICE";
        } else {
          maps[i].variable = qice;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case QNGRAUPEL:
        if (initial) {
          maps[i].name = "QNGRAUPEL";
          maps[i].out_name = "QNGRAUPEL";
        } else {
          maps[i].variable = qngraupel;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case QNICE:
        if (initial) {
          maps[i].name = "QNICE";
          maps[i].out_name = "QNICE";
        } else {
          maps[i].variable = qnice;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case QNRAIN:
        if (initial) {
          maps[i].name = "QNRAIN";
          maps[i].out_name = "QNRAIN";
        } else {
          maps[i].variable = qnrain;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case QNSNOW:
        if (initial) {
          maps[i].name = "QNSNOW";
          maps[i].out_name = "QNSNOW";
        } else {
          maps[i].variable = qnsnow;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case QRAIN:
        if (initial) {
          maps[i].name = "QRAIN";
          maps[i].out_name = "QRAIN";
        } else {
          maps[i].variable = qrain;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case QSNOW:
        if (initial) {
          maps[i].name = "QSNOW";
          maps[i].out_name = "QSNOW";
        } else {
          maps[i].variable = qsnow;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case QVAPOR:
        if (initial) {
          maps[i].name = "QVAPOR";
          maps[i].out_name = "QVAPOR";
        } else {
          maps[i].variable = qvapor;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case SH2O:
        if (initial) {
          maps[i].name = "SH2O";
          maps[i].out_name = "SH2O";
        } else {
          maps[i].variable = sh2o;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case SMCREL:
        if (initial) {
          maps[i].name = "SMCREL";
          maps[i].out_name = "SMCREL";
        } else {
          maps[i].variable = smcrel;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case SMOIS:
        if (initial) {
          maps[i].name = "SMOIS";
          maps[i].out_name = "SMOIS";
        } else {
          maps[i].variable = smois;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case T:
        if (initial) {
          maps[i].name = "T";
          maps[i].out_name = "PERT_T";
        } else {
          maps[i].variable = t;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case TSLB:
        if (initial) {
          maps[i].name = "TSLB";
          maps[i].out_name = "TSLB";
        } else {
          maps[i].variable = tslb;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case U:
        if (initial) {
          maps[i].name = "U";
          maps[i].out_name = "U";
        } else {
          maps[i].variable = u;
          maps[i].mass_variable = t;
          maps[i].longi = xlong_u;
          maps[i].lat = xlat_u;
          maps[i].mass_longi = xlong;
          maps[i].mass_lat = xlat;
        }
        break;
      case V:
        if (initial) {
          maps[i].name = "V";
          maps[i].out_name = "V";
        } else {
          maps[i].variable = v;
          maps[i].mass_variable = t;
          maps[i].longi = xlong_v;
          maps[i].lat = xlat_v;
          maps[i].mass_longi = xlong;
          maps[i].mass_lat = xlat;
        }
        break;
      case W:
        if (initial) {
          maps[i].name = "W";
          maps[i].out_name = "W";
        } else {
          maps[i].variable = w;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case PRESSURE:
        if (initial) {
          maps[i].name = "PRESSURE";
          maps[i].out_name = "PRESSURE";
        } else {
          maps[i].variable = pressure;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case COR_EAST:
        if (initial) {
          maps[i].name = "COR_EAST";
          maps[i].out_name = "COR_EAST";
        } else {
          maps[i].variable = cor_east;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case COR_NORTH:
        if (initial) {
          maps[i].name = "COR_NORTH";
          maps[i].out_name = "COR_NORTH";
        } else {
          maps[i].variable = cor_north;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case GEOPOTENTIAL:
        if (initial) {
          maps[i].name = "GEOPOTENTIAL";
          maps[i].out_name = "GEOPOT";
        } else {
          maps[i].variable = geopotential;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case COR_PARAM:
        if (initial) {
          maps[i].name = "COR_PARAM";
          maps[i].out_name = "COR_PARAM";
        } else {
          maps[i].variable = cor_param;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case ABS_VERT_VORT:
        if (initial) {
          maps[i].name = "ABS_VERT_VORT";
          maps[i].out_name = "ABS_VERT_VORT";
        } else {
          maps[i].variable = abs_vert_vort;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
      case REL_VERT_VORT:
        if (initial) {
          maps[i].name = "REL_VERT_VORT";
          maps[i].out_name = "REL_VERT_VORT";
        } else {
          maps[i].variable = rel_vert_vort;
          maps[i].longi = xlong;
          maps[i].lat = xlat;
        }
        break;
    }
  }
}

int get_flag(const char *name) {

  int flag = 0;
  for (int j = 0; j < NUM_VARIABLES; j++) {
    if (active_flags[j] != NULL) {
      char *str = (char *)active_flags[j];
      char dest[MAX_STRING_LENGTH];
      memset(dest, 0, sizeof(dest));
      strcpy(dest, name);
      strcat(dest, ":1");

      if (strcmp(str, dest) == 0) {
        flag = 1;
        break;
      }
    }
  }
  return flag;
}

void set_maps_active(map *maps) {

  fprintf(stdout, "Active variables: \n");
  for (int i = 0; i < NUM_VARIABLES; i++) {
    if (get_flag(maps[i].name)) {
      maps[i].active = true;
      printf("\t%s\n", maps[i].name);
    }
  }
}

// ----------------------------------------------------------------------------------------
// This routine checks dependency between variables, i.e. it makes sure that variables are
// computed in the right order if there is a dependency between them.
// It may seeem not really needed since the order to compute the variables is currenhtly
// hard coded. However this routine will be usefull when we will load what to compute from
// an exteral source in which the order may be arbitrary defined.
// NOTE: EXPERIMENTAL
// ----------------------------------------------------------------------------------------
void check_depedencies(map *maps) {

  if (!maps[COR_EAST].active && !maps[COR_NORTH].active && !maps[ABS_VERT_VORT].active &&
      !maps[GEOPOTENTIAL].active) {
        return; // Nothing to do
  }

  // Get the position in the map of U, V, PH, PHB and ABS_VERT_VORT
  int u_idx = -1;
  int v_idx = -1;
  int ph_idx = -1;
  int phb_idx = -1;
  for (int i = 0; i < NUM_VARIABLES; i++) {
    if (strcmp(maps[i].name, "U") == 0) u_idx = i;
    if (strcmp(maps[i].name, "V") == 0) v_idx = i;
    if (strcmp(maps[i].name, "PH") == 0) ph_idx = i;
    if (strcmp(maps[i].name, "PHB") == 0) phb_idx = i;
  }

  bool failure = false;
  if (maps[COR_EAST].active && v_idx < 0) failure = true;
  if (maps[COR_NORTH].active && u_idx < 0) failure = true;
  if (maps[ABS_VERT_VORT].active && (u_idx < 0 || v_idx < 0)) failure = true;
  if (maps[GEOPOTENTIAL].active && (ph_idx < 0 || phb_idx < 0)) failure = true;
  if (failure) {
    fprintf(stderr, "Dependency check failure due to wrong map. A dependent vartiable is computed but ",
                     "its needed variable is not defined in the map.\n");
  }

  for (int i = 0; i < NUM_VARIABLES; i++) {
    if (strcmp(maps[i].name, "COR_EAST") == 0 && maps[i].active) {
      if (i < v_idx) {
        fprintf(stderr, "ERROR: variable %s being defined too early in the maps so will be computed before V.\n",
                maps[i].name);
        exit(EXIT_FAILURE);
      }
    }

    if (strcmp(maps[i].name, "COR_NORTH") == 0 && maps[i].active) {
      if (i < u_idx) {
        fprintf(stderr, "ERROR: variable %s being defined too early in the maps so will be computed before U.\n",
                maps[i].name);
        exit(EXIT_FAILURE);
      }
    }

    if (strcmp(maps[i].name, "ABS_VERT_VORT") == 0 && maps[i].active) {
      if (i < u_idx) {
        fprintf(stderr, "ERROR: variable %s being defined too early in the maps so will be computed before U.\n",
                maps[i].name);
        exit(EXIT_FAILURE);
      }
      if (i < v_idx) {
        fprintf(stderr, "ERROR: variable %s being defined too early in the maps so will be computed before V.\n",
                maps[i].name);
        exit(EXIT_FAILURE);
      }
    }

    if (strcmp(maps[i].name, "GEOPOTENTIAL") == 0 && maps[i].active) {
      if (i < ph_idx) {
        fprintf(stderr, "ERROR: variable %s being defined too early in the maps so will be computed before PH.\n",
                maps[i].name);
        exit(EXIT_FAILURE);
      }
      if (i < phb_idx) {
        fprintf(stderr, "ERROR: variable %s being defined too early in the maps so will be computed before PHB.\n",
                maps[i].name);
        exit(EXIT_FAILURE);
      }
    }
  }
}

int check_maps(map *maps) {
  int status = 0;
  for (int i = 0; i < NUM_VARIABLES; i++) {
    if (maps[i].name == NULL) {
      fprintf(stderr, "Name field not defined in mapping - Error at loop index: %d.\n", i);
      status = -1;
    }
  }

  for (int i = 0; i < NUM_VARIABLES; i++) {
    if (strcmp(maps[i].name, "XLAT")    == 0 || strcmp(maps[i].name, "XLAT_U")  == 0 ||
        strcmp(maps[i].name, "XLAT_V")  == 0 || strcmp(maps[i].name, "XLONG")   == 0 ||
        strcmp(maps[i].name, "XLONG_U") == 0 || strcmp(maps[i].name, "XLONG_V") == 0)
           continue;

    if (maps[i].out_name == NULL) {
      fprintf(stderr, "Output name field not defined in mapping for variable: %s.\n", maps[i].name);
      status = -1;
    }
  }
  return status;
}

void get_files_from_dir(const char *directory, char files[][MAX_STRING_LENGTH], uint * num_files) {

  DIR *d;
  struct dirent *dir;
  d = opendir(directory);
  if (d) {
    *num_files = 0;
    while ((dir = readdir(d)) != NULL) {
      strcpy(files[*num_files], dir->d_name);
      (*num_files)++;
    }
  }
  closedir(d);
}

int load_variable(int ncid, const char *var_name, tensor * t, bool active, bool used) {

  int retval = 0;
  int varid;

  // The special case of the full pressure, coriolis east and coriolus north,
  // the full dry mass, the geopotential, coriolis parameter and absolute
  // vertival vorticity
  // Will be computed later after first loading the needed variable to do so
  if (strcmp(var_name, "PRESSURE") == 0) {
    return retval;
  }
  if (strcmp(var_name, "COR_EAST") == 0) {
    return retval;
  }
  if (strcmp(var_name, "COR_NORTH") == 0) {
    return retval;
  }
  if (strcmp(var_name, "DRY_MASS") == 0) {
    return retval;
  }
  if (strcmp(var_name, "GEOPOTENTIAL") == 0) {
    return retval;
  }
  if (strcmp(var_name, "COR_PARAM") == 0) {
    return retval;
  }
  if (strcmp(var_name, "ABS_VERT_VORT") == 0) {
    return retval;
  }
  if (strcmp(var_name, "REL_VERT_VORT") == 0) {
    return retval;
  }
  if (!active && !used) {
    return retval;
  }

  retval = nc_inq_varid(ncid, var_name, &varid);
  retval = nc_get_var_float(ncid, varid, t->val);

  return retval;
}

void set_visual_path(char *str) {

  strncpy(str, WORKDIR, MAX_STRING_LENGTH);
  strncat(str, "/VISUAL", strlen("/VISUAL"));
}

void set_nn_path(char *str) {

  strncpy(str, WORKDIR, MAX_STRING_LENGTH);
  strncat(str, "/NN", strlen("/NN"));
}

bool is_directory(char *dir) {

  struct stat st = {0};
  if (stat(dir, &st) == -1) {
    return false;
  } else return true;
}

int create_directory(char *dir) {

  int err;
  if (is_directory(dir)) {
    return 0;
  } else {
    err = mkdir(dir, 0700);
  }
  return err;
}

void convert_to_string(char *str, int value) {

  int length = snprintf( NULL, 0, "%d", value);
  snprintf(str, length + 1, "%d", value);
}

void set_path(char dir[], char *run, const char *out_name) {
  strncat(dir, "/", strlen("/"));
  strncat(dir, run, strlen(run));

  strncat(dir, "/", strlen("/"));
  strncat(dir, out_name, strlen(out_name));
}

void get_horizontal_dims(int idx, uint *nx, uint *ny) {
  if (maps[idx].variable->rank > 3) {
    *ny = maps[idx].variable->shape[2];
    *nx = maps[idx].variable->shape[3];
  } else {
    *ny = maps[idx].variable->shape[1];
    *nx = maps[idx].variable->shape[2];
  }
}

void write_visual_to_file(FILE *file, map *maps, int idx, int z) {

  uint nx = 0;
  uint ny = 0;

  get_horizontal_dims(idx, &nx, &ny);

  fprintf(file, "longitude,latitude,%s\n", maps[idx].out_name);

  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
        float longi = maps[idx].longi->val[(y*nx)+x];
        float lat   = maps[idx].lat->val[(y*nx)+x];
        float val   = maps[idx].variable->val[(z*(ny*nx))+((y*nx)+x)];
        fprintf(file, "%f,%f,%f\n", longi, lat, val);
      }
    }
}

void write_nn_to_file(FILE *file, map *maps, int idx, int z, feature_scaling_pt feature_scaling_func) {

  uint nx = 0;
  uint ny = 0;

  get_horizontal_dims(idx, &nx, &ny);

  if (strcmp(maps[idx].name, "ZNU") == 0 || strcmp(maps[idx].name, "ZNW") == 0) {
    fprintf(file, "%s\n", maps[idx].out_name);
    float val = maps[idx].variable->val[z];
    fprintf(file, "%f\n", val);
    return;
  }

  fprintf(file, "longitude,latitude,%s\n", maps[idx].out_name);

  float longi[ny*nx];
  float lat[ny*nx];
  float val[ny*nx];

  bool new_call = true;
  int i = 0;
  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
        longi[i] = feature_scaling_func(maps[idx].longi->val[(y*nx)+x],
                                        maps[idx].longi->val, ny*nx, &new_call);
        i++;
    }
  }

  new_call = true;
  i = 0;
  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
        lat[i] = feature_scaling_func(maps[idx].lat->val[(y*nx)+x],
                                      maps[idx].lat->val, ny*nx, &new_call);
        i++;
    }
  }

  new_call = true;
  i = 0;
  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
        val[i] = feature_scaling_func(maps[idx].variable->val[(z*(ny*nx))+((y*nx)+x)],
                                      maps[idx].variable->val+(z*(ny*nx)), ny*nx, &new_call);
        i++;
    }
  }

  i = 0;
  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
        fprintf(file, "%f,%f,%f\n", longi[i], lat[i], val[i]);
        i++;
    }
  }
}

void get_neighbors_val_horiz(velo_grid *h_velo_u_grid, velo_grid *h_velo_v_grid, map *maps,
      uint NY_STAG, uint NX_STAG, uint NY, uint NX, uint num_support_points, char *data_name, int *z) {

  if (data_name == NULL) {
    fprintf(stderr, "Data name missing in routing get_neighbors_val_horiz().\n");
  }

  static bool check = false;
  if (strcmp(data_name, "velocities") == 0) {
    if (!check) {
      if (z == NULL) {
        fprintf(stderr, "z coordinate missing in routing get_neighbors_val_horiz().\n");
      }
      check = true;
    }
  }

  int offset1, offset2;

  if (strcmp(data_name, "coordinates") == 0) {
    offset1 = 0;
    offset2 = 0;
  } else if (strcmp(data_name, "velocities") == 0) {
    offset1 = *z*NY*NX_STAG;
    offset2 = *z*NY_STAG*NX;
  } else {
    fprintf(stderr, "Incorrect data name.\n");
  }

  // Get the coordinates/velocities of the neighbors of each point in the grid NY x NX
  // The neighbors are taken from the U and V grids
  // Only for two supporting points
  if (strcmp(data_name, "coordinates") == 0) {
    for (int i = 0; i < num_support_points; i++) {
      int idx = 0;
      for (int j = 0; j < NY; j++) {
        for(int k = 0; k < NX; k++) {
          switch (i) {
            case 0: // Left point
              h_velo_u_grid->x[((NY*NX)*i)+idx] = maps[U].longi->val[offset1+(NX_STAG*j)+k];
              h_velo_u_grid->y[((NY*NX)*i)+idx] = maps[U].lat->val[offset1+(NX_STAG*j)+k];

              h_velo_v_grid->x[((NY*NX)*i)+idx] = maps[V].longi->val[offset2+(NX*j)+k];
              h_velo_v_grid->y[((NY*NX)*i)+idx] = maps[V].lat->val[offset2+(NX*j)+k];
              break;
            case 1: // Right point
              h_velo_u_grid->x[((NY*NX)*i)+idx] = maps[U].longi->val[offset1+(NX_STAG*j)+k+1];
              h_velo_u_grid->y[((NY*NX)*i)+idx] = maps[U].lat->val[offset1+(NX_STAG*j)+k+1];

              h_velo_v_grid->x[((NY*NX)*i)+idx] = maps[V].longi->val[offset2+(NX*(j+1))+k];
              h_velo_v_grid->y[((NY*NX)*i)+idx] = maps[V].lat->val[offset2+(NX*(j+1))+k];
            break;
          }
          idx++;
        }
      }
    }
  } else if (strcmp(data_name, "velocities") == 0) {
    for (int i = 0; i < num_support_points; i++) {
      int idx = 0;
      for (int j = 0; j < NY; j++) {
        for(int k = 0; k < NX; k++) {
          switch (i) {
            case 0: // Left point
              h_velo_u_grid->val[((NY*NX)*i)+idx] = maps[U].variable->val[offset1+(NX_STAG*j)+k];
              h_velo_v_grid->val[((NY*NX)*i)+idx] = maps[V].variable->val[offset2+(NX*j)+k];
              break;
            case 1: // Right point
              h_velo_u_grid->val[((NY*NX)*i)+idx] = maps[U].variable->val[offset1+(NX_STAG*j)+k+1];
              h_velo_v_grid->val[((NY*NX)*i)+idx] = maps[V].variable->val[offset2+(NX*(j+1))+k];
            break;
          }
          idx++;
        }
      }
    }
  }
}

void get_neighbors_values_vert(velo_grid *h_grid, map *maps, uint var, uint NY, uint NX, int z,
  uint num_support_points) {

  // Get the coordinates of the neighbors in the vertical direction in the
  // grid NY x NX. The neighbors are taken from the W grid
  // Only for two supporting points
  for (int i = 0; i < num_support_points; i++) {
    int idx = 0;
    for (int j = 0; j < NY; j++) {
      for(int k = 0; k < NX; k++) {
        h_grid->val[((NY*NX)*i)+idx] = maps[var].variable->val[((z+i)*NY*NX)+(NX*j)+k];
        idx++;
      }
    }
  }
}

void staggered_var_scaling(float *u, float *v, float *w, float *longi, float *lat,
     uint NX, uint NY, map *maps, mass_grid *h_mass_grid, int idx, bool scale_u, bool scale_v, bool scale_w,
     bool scale_phb, bool scale_ph, feature_scaling_pt feature_scaling_func) {


  tensor *long_pt = NULL;
  tensor *lat_pt = NULL;

  if (scale_u || scale_v) {
    long_pt = maps[idx].mass_longi;
    lat_pt = maps[idx].mass_lat;
  } else if (scale_w || scale_phb || scale_ph) {
    long_pt = maps[idx].longi;
    lat_pt = maps[idx].lat;
  } else {
    fprintf(stderr, "Which variable to scale is not given.\n");
    exit(EXIT_FAILURE);
  }

  bool  new_call = true;
  int i = 0;
  for (int y = 0; y < NY; y++) {
    for (int x = 0; x < NX; x++) {
      longi[i] = feature_scaling_func(long_pt->val[(y*long_pt->shape[2])+x], long_pt->val, NY*NX, &new_call);
      i++;
    }
  }

  new_call = true;
  i = 0;
  for (int y = 0; y < NY; y++) {
    for (int x = 0; x < NX; x++) {
      lat[i] = feature_scaling_func(lat_pt->val[(y*lat_pt->shape[2])+x], lat_pt->val, NY*NX, &new_call);
      i++;
    }
  }

  if (scale_u) {
    if (u == NULL) {
      fprintf(stderr, "Scale U but null pointer.\n");
      exit(EXIT_FAILURE);
    }
    new_call = true;
    i = 0;
    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        u[i] = feature_scaling_func(h_mass_grid->u[(y*maps[idx].mass_variable->shape[3])+x],
                    h_mass_grid->u, NY*NX, &new_call);
        i++;
      }
    }
  }

  if (scale_v) {
    if (v == NULL) {
      fprintf(stderr, "Scale V but null pointer.\n");
      exit(EXIT_FAILURE);
    }
    new_call = true;
    i = 0;
    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        v[i] = feature_scaling_func(h_mass_grid->v[(y*maps[idx].mass_variable->shape[3])+x],
                    h_mass_grid->v, NY*NX, &new_call);
        i++;
      }
    }
  }

  if (scale_w) {
    if (w == NULL) {
      fprintf(stderr, "Scale W but null pointer.\n");
      exit(EXIT_FAILURE);
    }
    new_call = true;
    i = 0;
    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        w[i] = feature_scaling_func(h_mass_grid->w[(y*maps[idx].variable->shape[3])+x],
                    h_mass_grid->w, NY*NX, &new_call);
      }
    }
  }

  if (scale_phb) {
    if (w == NULL) {
      fprintf(stderr, "Scale base geopotential but null pointer.\n");
      exit(EXIT_FAILURE);
    }
    new_call = true;
    i = 0;
    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        w[i] = feature_scaling_func(h_mass_grid->phb[(y*maps[idx].variable->shape[3])+x],
                    h_mass_grid->phb, NY*NX, &new_call);
      }
    }
  }

  if (scale_ph) {
    if (w == NULL) {
      fprintf(stderr, "Scale perturbation geopotential but null pointer.\n");
      exit(EXIT_FAILURE);
    }
    new_call = true;
    i = 0;
    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        w[i] = feature_scaling_func(h_mass_grid->ph[(y*maps[idx].variable->shape[3])+x],
                    h_mass_grid->ph, NY*NX, &new_call);
      }
    }
  }
}

void vorticity_scaling(float *vort,  float *longi, float *lat, uint NX, uint NY, map *maps, fd_container *vorticity,

  int idx, feature_scaling_pt feature_scaling_func) {
   bool  new_call = true;
    int i = 0;
    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        longi[i] = feature_scaling_func(maps[idx].longi->val[(y*NX)+x], maps[idx].longi->val, NY*NX, &new_call);
        i++;
      }
    }

    new_call = true;
    i = 0;
    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        lat[i] = feature_scaling_func(maps[idx].lat->val[(y*NX)+x], maps[idx].lat->val, NY*NX, &new_call);
        i++;
      }
    }

    new_call = true;
    i = 0;
    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        vort[i] = feature_scaling_func(vorticity->val[(y*NX)+x], vorticity->val, NY*NX, &new_call);
        i++;
      }
    }
}

#ifdef __NVCC__
void interpolate_wind_velo_horiz(map *maps, int idx, int z, char file[][MAX_STRING_LENGTH], char file_nn[][MAX_STRING_LENGTH],
                          dim3 block, dim3 grid, int grid_type, feature_scaling_pt feature_scaling_func) {
#else
void interpolate_wind_velo_horiz(map *maps, int idx, int z, char file[][MAX_STRING_LENGTH], char file_nn[][MAX_STRING_LENGTH],
                         float **buffer, int grid_type, feature_scaling_pt feature_scaling_func) {
#endif

    FILE *f = fopen(file[0], "w");
    fprintf(f, "longitude,latitude,%s\n", maps[idx].out_name);

    FILE *f_nn = fopen(file_nn[0], "w");
    fprintf(f_nn, "longitude,latitude,%s\n", maps[idx].out_name);
#ifdef __NVCC__
    FILE *f_bis = fopen(file[1], "w");
    fprintf(f_bis, "longitude,latitude,%s\n", maps[V].out_name);

    FILE *f_nn_bis = fopen(file_nn[1], "w");
    fprintf(f_nn_bis, "longitude,latitude,%s\n", maps[V].out_name);
#endif

#ifdef __NVCC__
    fprintf(stdout, "( ------ Parallel interpolate <<<%d, %d>>> %s %s at layer %d.\n", grid.x, block.x,
            maps[idx].name, maps[V].name, z);
#else
    fprintf(stdout, "( ------ Interpolate %s at layer %d\n", maps[idx].name, z);
#endif

    if (grid_type == STRUCTURED) {
      // Get the neighbors values of the mass points
      char *data_name = (char *)"velocities";
      double nv = cpu_second();
      get_neighbors_val_horiz(h_velo_u_grid, h_velo_v_grid, maps, NY_STAG, NX_STAG, NY, NX,
        NUM_SUPPORTING_POINTS_HORIZ, data_name, &z);
      fprintf(stdout, "Time to get neighbors values (horiz): %f sec.\n", cpu_second() - nv);
    }

#ifdef __NVCC__

    if (grid_type == STRUCTURED) {
      {
        float *v[1];
        cudaMemcpy(&(v[0]), &(d_velo_u_grid[0].val), sizeof(float *), cudaMemcpyDeviceToHost);
        cudaMemcpy(v[0], h_velo_u_grid->val, (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float), cudaMemcpyHostToDevice);
      }
      {
        float *vv[1];
        cudaMemcpy(&(vv[0]), &(d_velo_v_grid[0].val), sizeof(float *), cudaMemcpyDeviceToHost);
        cudaMemcpy(vv[0], h_velo_v_grid->val, (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float), cudaMemcpyHostToDevice);
      }
    }

    double i_start = cpu_second();
    if (grid_type == STRUCTURED) {
        gpu_radially_interpolate_structured_horiz<<<grid, block>>>(d_velo_u_grid, d_velo_v_grid, d_mass_grid,
                                    NY, NX, NUM_SUPPORTING_POINTS_HORIZ, 2.0f);
    } else {
        gpu_radially_interpolate_unstructured <<< grid, block >>>(d_velo_u_grid, d_velo_v_grid, d_mass_grid,
                                    NY_STAG, NX_STAG, NY, NX, z, 2, 4, 2.0f);
    }

    cudaDeviceSynchronize();
    double i_elaps = cpu_second() - i_start;
    fprintf(stdout, ">>>>>>>>>>>> elapsed (compute): %f sec.\n", i_elaps);

    i_start = cpu_second();
    float *u_mass[1];
    cudaMemcpy(&(u_mass[0]), &(d_mass_grid[0].u), sizeof(float *), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mass_grid->u, u_mass[0], (NY*NX)*sizeof(float), cudaMemcpyDeviceToHost);

    float *v_mass[1];
    cudaMemcpy(&(v_mass[0]), &(d_mass_grid[0].v), sizeof(float *), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mass_grid->v, v_mass[0], (NY*NX)*sizeof(float), cudaMemcpyDeviceToHost);
    i_elaps = cpu_second() - i_start;
    fprintf(stdout, ">>>>>>>>>>>> elapsed (data copy): %f sec.\n", i_elaps);

    i_start = cpu_second();
    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        float longi = maps[idx].mass_longi->val[(y*maps[idx].mass_longi->shape[2])+x];
        float lat   = maps[idx].mass_lat->val[(y*maps[idx].mass_lat->shape[2])+x];
        float u_val = h_mass_grid->u[(y*maps[idx].mass_variable->shape[3])+x];
        float v_val = h_mass_grid->v[(y*maps[idx].mass_variable->shape[3])+x];
        fprintf(f, "%f,%f,%f\n", longi, lat, u_val);
        fprintf(f_bis, "%f,%f,%f\n", longi, lat, v_val);
      }
    }

    float longi[NY*NX];
    float lat[NY*NX];
    float u_nn_val[NY*NX];
    float v_nn_val[NY*NX];
    memset(longi, 0.0f, sizeof(longi));
    memset(lat, 0.0f, sizeof(lat));
    memset(u_nn_val, 0.0f, sizeof(u_nn_val));
    memset(v_nn_val, 0.0f, sizeof(v_nn_val));

    staggered_var_scaling(u_nn_val, v_nn_val, NULL, longi, lat, NX, NY, maps, h_mass_grid, idx, true, true, false,
                     false, false, feature_scaling_func);

    int i = 0;
    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        fprintf(f_nn, "%f,%f,%f\n", longi[i], lat[i], u_nn_val[i]);
        fprintf(f_nn_bis, "%f,%f,%f\n", longi[i], lat[i], v_nn_val[i]);
        i++;
      }
    }

    i_elaps = cpu_second() -  i_start;
    fprintf(stdout, ">>>>>>>>>>>> elapsed (write file): %f sec.\n", i_elaps);

    // If the coriolis force is needed, store here the interpolated
    // velocities that will be needed later
    if (maps[COR_EAST].active) {
      float *coriolis = maps[COR_EAST].variable->val;

      uint nx = 0;
      uint ny = 0;
      get_horizontal_dims(COR_EAST, &nx, &ny);

      for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
          float v_val = h_mass_grid->v[(y*nx)+x];
          coriolis[(z*(ny*nx))+((y*nx)+x)] = v_val;
        }
      }
    }

    if (maps[COR_NORTH].active) {
      float *coriolis = maps[COR_NORTH].variable->val;

      uint nx = 0;
      uint ny = 0;
      get_horizontal_dims(COR_NORTH, &nx, &ny);

      for (int y = 0; y < ny; y++) {
        for (int x = 0; x <nx; x++) {
          float u_val = h_mass_grid->u[(y*nx)+x];
          coriolis[(z*(ny*nx))+((y*nx)+x)] = u_val;
        }
      }
    }

    // If the absolute/relative vertical vorticity is computed, store here
    // the interpolated velocities
    if (maps[ABS_VERT_VORT].active) {
      uint nx = 0;
      uint ny = 0;
      get_horizontal_dims(ABS_VERT_VORT, &nx, &ny);

      for (int y = 0; y < ny; y++) {
        for (int x = 0; x <nx; x++) {
          float u_val = h_mass_grid->u[(y*nx)+x];
          float v_val = h_mass_grid->v[(y*nx)+x];
          maps[ABS_VERT_VORT].buffer1[(z*(ny*nx))+((y*nx)+x)] = u_val;
          maps[ABS_VERT_VORT].buffer2[(z*(ny*nx))+((y*nx)+x)] = v_val;
        }
      }
    }
    if (maps[REL_VERT_VORT].active) {
      uint nx = 0;
      uint ny = 0;
      get_horizontal_dims(REL_VERT_VORT, &nx, &ny);

      for (int y = 0; y < ny; y++) {
        for (int x = 0; x <nx; x++) {
          float u_val = h_mass_grid->u[(y*nx)+x];
          float v_val = h_mass_grid->v[(y*nx)+x];
          maps[REL_VERT_VORT].buffer1[(z*(ny*nx))+((y*nx)+x)] = u_val;
          maps[REL_VERT_VORT].buffer2[(z*(ny*nx))+((y*nx)+x)] = v_val;
        }
      }
    }

#else
    int num_data;
    int dim = 2;
    int directions[2] = {1,2};
    bool verbose = true;
    bool reinitiate;

    if (grid_type == UNSTRUCTURED) {
      if (z > 0) {
        reinitiate = false;
      } else {
        reinitiate = true;
      }

      num_data = maps[idx].variable->shape[2] * maps[idx].variable->shape[3];
      int i = 0;
      for (int y = 0; y < maps[idx].variable->shape[2]; y++) {
        for (int x = 0; x < maps[idx].variable->shape[3]; x++) {
          buffer[i][0] = maps[idx].longi->val[(y*maps[idx].longi->shape[2])+x];
          buffer[i][1] = maps[idx].lat->val[(y*maps[idx].lat->shape[2])+x];
          buffer[i][2] = maps[idx].variable->val[(z*(maps[idx].variable->shape[2]*maps[idx].variable->shape[3]))
          +((y*maps[idx].variable->shape[3])+x)];
          i++;
        }
      }
    }

    int grid_idx = 0;
    velo_grid *pt_to_grid = NULL;
    if (grid_type == STRUCTURED) {
      if(strcmp(maps[idx].name, "U") == 0) {
        pt_to_grid = h_velo_u_grid;
      } else {
        pt_to_grid = h_velo_v_grid;
      }
    }

    uint ny = maps[idx].mass_variable->shape[2];
    uint nx = maps[idx].mass_variable->shape[3];
    float interpo[ny*nx];
    uint i = 0;
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        float longi = maps[idx].mass_longi->val[(y*maps[idx].mass_longi->shape[2])+x];
        float lat   = maps[idx].mass_lat->val[(y*maps[idx].mass_lat->shape[2])+x];
        if (grid_type == STRUCTURED) {
          interpol[i] = cpu_radially_interpolate_structured(pt_to_grid, &longi, &lat,
                            grid_idx, NY, NX, NUM_SUPPORTING_POINTS_HORIZ, 2.0f);
            grid_idx++;
        } else {
          interpol[i] = cpu_radially_interpolate_unstructured(buffer, &longi, &lat, NULL, num_data, dim, directions,
                            2.0f, reinitiate, 4, &verbose);
        }
        i++;
      }
    }

    bool new_call = true;
    i = 0;
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        float longi = maps[idx].mass_longi->val[(y*maps[idx].mass_longi->shape[2])+x];
        float lat   = maps[idx].mass_lat->val[(y*maps[idx].mass_lat->shape[2])+x];
        float val   = feature_scaling_func(interpol[i], interpol, ny*nx, &new_call);
        fprintf(f,  "%f,%f,%f\n", longi, lat, interpol[i]);
        fprintf(f_nn, "%f,%f,%f\n", longi, lat, val);
        i++;
      }
    }

    if (maps[COR_EAST].active || maps[COR_NORTH].active) {
      float *coriolis = NULL;

      uint nx = 0;
      uint ny = 0;

      if (strcmp(maps[idx].name, "U") == 0) {
        coriolis = maps[COR_NORTH].variable->val;
        get_horizontal_dims(COR_NORTH, &nx, &ny);
      } else {
        coriolis = maps[COR_EAST].variable->val;
        get_horizontal_dims(COR_EAST, &nx, &ny);
      }

      int i = 0;
      for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
          coriolis[(z*(ny*nx))+((y*nx)+x)] = interpol[i];
          i++;
        }
      }
    }

#endif

    fclose(f);
    fclose(f_nn);
#ifdef __NVCC__
    fclose(f_bis);
    fclose(f_nn_bis);
#endif
}

#ifdef __NVCC__
void interpolate_var_vert(velo_grid *h_var_grid, velo_grid *d_var_grid, float *mass_var, map *maps, int idx,
  int z, char file[][MAX_STRING_LENGTH], char file_nn[][MAX_STRING_LENGTH], dim3 block, dim3 grid,
  int grid_type, feature_scaling_pt feature_scaling_func) {
#else
void interpolate_var_vert(velo_grid *h_var_grid, map *maps, int idx, int z, char file[][MAX_STRING_LENGTH],
  char file_nn[][MAX_STRING_LENGTH], float **buffer, int grid_type, feature_scaling_pt feature_scaling_func) {
#endif

    FILE *f = fopen(file[0], "w");
    fprintf(f, "longitude,latitude,%s\n", maps[idx].out_name);

    FILE *f_nn = fopen(file_nn[0], "w");
    fprintf(f_nn, "longitude,latitude,%s\n", maps[idx].out_name);

    fprintf(stdout, "( ------ Interpolate %s at layer %d\n", maps[idx].name, z);

    if (grid_type == STRUCTURED) {
      // Get the neighbors values of the mass points
      double nv = cpu_second();
      get_neighbors_values_vert(h_var_grid, maps, idx, NY, NX, z, NUM_SUPPORTING_POINTS_VERT);
      fprintf(stdout, "Time to get neighbors values (vert): %f sec.\n", cpu_second() - nv);
    }

#ifdef __NVCC__

    if (grid_type == STRUCTURED) {
      {
        float *v[1];
        cudaMemcpy(&(v[0]), &(d_var_grid[0].val), sizeof(float *), cudaMemcpyDeviceToHost);
        cudaMemcpy(v[0], h_var_grid->val, (NY*NX*NUM_SUPPORTING_POINTS_VERT)*sizeof(float), cudaMemcpyHostToDevice);
      }
    }

    double i_start = cpu_second();
    if (grid_type == STRUCTURED) {
        float z_level;
        float z_level_stag_under;
        float z_level_stag_above;
        z_level = maps[ZNU].variable->val[(NZ-1)-z];
        z_level_stag_under = maps[ZNW].variable->val[(NZ_STAG-1)-z];
        z_level_stag_above = maps[ZNW].variable->val[(NZ_STAG-1)-z-1];

        gpu_radially_interpolate_structured_vert<<<grid, block>>>(d_var_grid, d_mass_grid,
                                    NY, NX, z_level, z_level_stag_under, z_level_stag_above,
                                    NUM_SUPPORTING_POINTS_VERT, 2.0f);
    } else {
        // TODO: Need to be implemented if we need that
    }

    cudaDeviceSynchronize();
    double i_elaps = cpu_second() - i_start;
    fprintf(stdout, ">>>>>>>>>>>> elapsed (compute): %f sec.\n", i_elaps);

    i_start = cpu_second();
    float *w_mass[1];
    cudaMemcpy(&(w_mass[0]), &(d_mass_grid[0].w), sizeof(float *), cudaMemcpyDeviceToHost);
    cudaMemcpy(mass_var, w_mass[0], (NY*NX)*sizeof(float), cudaMemcpyDeviceToHost);

    i_elaps = cpu_second() - i_start;
    fprintf(stdout, ">>>>>>>>>>>> elapsed (data copy): %f sec.\n", i_elaps);

    i_start = cpu_second();
    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        float longi = maps[idx].longi->val[(y*maps[idx].longi->shape[2])+x];
        float lat   = maps[idx].lat->val[(y*maps[idx].lat->shape[2])+x];
        float w_val = mass_var[(y*maps[idx].variable->shape[3])+x];
        fprintf(f, "%f,%f,%f\n", longi, lat, w_val);
      }
    }

    float longi[NY*NX];
    float lat[NY*NX];
    float w_nn_val[NY*NX];
    memset(longi, 0.0f, sizeof(longi));
    memset(lat, 0.0f, sizeof(lat));
    memset(w_nn_val, 0.0f, sizeof(w_nn_val));

    if (strcmp(maps[idx].name, "W") == 0) {
      staggered_var_scaling(NULL, NULL, w_nn_val, longi, lat, NX, NY, maps, h_mass_grid, idx, false, false, true,
                    false, false, feature_scaling_func);
    } else if (strcmp(maps[idx].name, "PHB") == 0) {
      staggered_var_scaling(NULL, NULL, w_nn_val, longi, lat, NX, NY, maps, h_mass_grid, idx, false, false, false,
                    true, false, feature_scaling_func);
    } else if (strcmp(maps[idx].name, "PH") == 0) {
      staggered_var_scaling(NULL, NULL, w_nn_val, longi, lat, NX, NY, maps, h_mass_grid, idx, false, false, false,
                    false, true, feature_scaling_func);
    }

    int i = 0;
    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        fprintf(f_nn, "%f,%f,%f\n", longi[i], lat[i], w_nn_val[i]);
        i++;
      }
    }

    i_elaps = cpu_second() -  i_start;
    fprintf(stdout, ">>>>>>>>>>>> elapsed (write file): %f sec.\n", i_elaps);

    // If the full geopotential is needed, store here the interpolated
    // base and perturbation geopotential that will be needed later
    if (maps[GEOPOTENTIAL].active) {
      if (strcmp(maps[idx].name, "PHB") == 0 || strcmp(maps[idx].name, "PH") == 0) {
        float *pt = NULL;
        uint nx = 0;
        uint ny = 0;

        if (strcmp(maps[idx].name, "PHB") == 0) {
          pt = maps[PHB].variable->val;
          get_horizontal_dims(PHB, &nx, &ny);
        } else {
          pt = maps[PH].variable->val;
          get_horizontal_dims(PH, &nx, &ny);
        }

        for (int y = 0; y < ny; y++) {
          for (int x = 0; x < nx; x++) {
            pt[(z*(ny*nx))+((y*nx)+x)] = mass_var[(y*nx)+x];
          }
        }
      }
    }

#else

    if (grid_type == UNSTRUCTURED) {
      // TODO: Need to be implemented if we need that
    }

    int grid_idx = 0;
    velo_grid *pt_to_grid = NULL;
    if (grid_type == STRUCTURED) {
      pt_to_grid = h_var_grid;
    }

    uint ny = maps[idx].variable->shape[2];
    uint nx = maps[idx].variable->shape[3];
    float interpo[ny*nx];
    uint i = 0;
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        float longi = maps[idx].longi->val[(y*maps[idx].longi->shape[2])+x];
        float lat   = maps[idx].lat->val[(y*maps[idx].lat->shape[2])+x];
        if (grid_type == STRUCTURED) {
          interpol[i] = cpu_radially_interpolate_structured_vert(pt_to_grid, &longi, &lat,
                            grid_idx, NY, NX, NUM_SUPPORTING_POINTS_VERT, 2.0f);
            grid_idx++;
        } else {
          // TODO: Need to be implemented if we need that
        }
        i++;
      }
    }

    bool new_call = true;
    i = 0;
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        float longi = maps[idx].longi->val[(y*maps[idx].longi->shape[2])+x];
        float lat   = maps[idx].lat->val[(y*maps[idx].lat->shape[2])+x];
        float val   = feature_scaling_func(interpol[i], interpol, ny*nx, &new_call);
        fprintf(f,  "%f,%f,%f\n", longi, lat, interpol[i]);
        fprintf(f_nn, "%f,%f,%f\n", longi, lat, val);
        i++;
      }
    }

#endif

    fclose(f);
    fclose(f_nn);
}

#ifdef __NVCC__
void compute_vert_vorticity(fd_container *h_vorticity, fd_container *d_vorticity, map *maps, int idx,
  int z, char file[][MAX_STRING_LENGTH], char file_nn[][MAX_STRING_LENGTH], dim3 block, dim3 grid,
  int grid_type, feature_scaling_pt feature_scaling_func, bool absolute) {
#else
void compute_vert_vorticity(fd__container *h_vorticity, map *maps, int idx, int z, char file[][MAX_STRING_LENGTH],
  char file_nn[][MAX_STRING_LENGTH], float **buffer, int grid_typeinterpolate_var_vert,
  feature_scaling_pt feature_scaling_func, bool absolute) {
#endif

    if (grid_type == UNSTRUCTURED) {
      fprintf(stderr, "Finite difference approximation not for unstructured mesh.\n");
    }

    FILE *f = fopen(file[0], "w");
    fprintf(f, "longitude,latitude,%s\n", maps[idx].out_name);

    FILE *f_nn = fopen(file_nn[0], "w");
    fprintf(f_nn, "longitude,latitude,%s\n", maps[idx].out_name);

    fprintf(stdout, "( ------ Compute %s at layer %d\n", maps[idx].name, z);

    // Get the stencils values
    get_stencils_values(domain_tags, h_vorticity, maps[idx].buffer2, maps[idx].buffer1, NY, NX, z);

#ifdef __NVCC__
    {
      float *v[1];
      cudaMemcpy(&(v[0]), &(d_vorticity[0].stencils_val), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaMemcpy(v[0], h_vorticity->stencils_val, (6*NY*NX)*sizeof(float), cudaMemcpyHostToDevice);
    }

    double i_start = cpu_second();
    if (absolute) {
      gpu_compute_abs_vert_vort<<<grid, block>>>(d_vorticity, NY, NX, dy, dx, VORTICITY_SCALING);
    } else {
      gpu_compute_rel_vert_vort<<<grid, block>>>(d_vorticity, NY, NX, dy, dx, VORTICITY_SCALING);
    }

    cudaDeviceSynchronize();
    double i_elaps = cpu_second() - i_start;
    fprintf(stdout, ">>>>>>>>>>>> elapsed (compute): %f sec.\n", i_elaps);

    i_start = cpu_second();
    {
      float *vort[1];
      cudaMemcpy(&(vort[0]), &(d_vorticity[0].val), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_vorticity->val, vort[0], (NY*NX)*sizeof(float), cudaMemcpyDeviceToHost);
    }
    i_elaps = cpu_second() - i_start;
    fprintf(stdout, ">>>>>>>>>>>> elapsed (data copy): %f sec.\n", i_elaps);

    i_start = cpu_second();
    for (int y = 0; y < NY; y++) {
     for (int x = 0; x < NX; x++) {
       float longi = maps[idx].longi->val[(y*NX)+x];
       float lat   = maps[idx].lat->val[(y*NX)+x];
       float vort  = h_vorticity->val[(y*NX)+x];
       fprintf(f, "%f,%f,%f\n", longi, lat, vort);
     }
    }

    float longi[NY*NX];
    float lat[NY*NX];
    float vort_nn[NY*NX];
    memset(longi, 0.0f, sizeof(longi));
    memset(lat, 0.0f, sizeof(lat));
    memset(vort_nn, 0.0f, sizeof(vort_nn));

    vorticity_scaling(vort_nn, longi, lat, NX, NY, maps, h_vorticity, idx, feature_scaling_func);

    int i = 0;
    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        fprintf(f_nn, "%f,%f,%f\n", longi[i], lat[i], vort_nn[i]);
        i++;
      }
    }

    i_elaps = cpu_second() -  i_start;
    fprintf(stdout, ">>>>>>>>>>>> elapsed (write file): %f sec.\n", i_elaps);
#else
  // Not implemented yet
#endif

  fclose(f);
  fclose(f_nn);
}

//---------------------------------------------------------------------------------------------------------------------
//--------------------------------------------- Data write routine ----------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------
int write_data(map *maps, int idx, char *run, bool no_interpol_out, int grid_type,
               feature_scaling_pt feature_scaling_func) {

#ifndef __NVCC__
  float **buffer = NULL;
  uint buffer_size = 0;
#endif

  if(!maps[idx].active) return 0;

#ifdef __NVCC__
  if (strcmp(maps[idx].name, "V") == 0) return 0;
#endif

#ifndef __NVCC__
  if (strcmp(maps[idx].name, "U") == 0 || strcmp(maps[idx].name, "V") == 0) {
    if (grid_type == UNSTRUCTURED) {
      buffer_size = maps[idx].variable->shape[2] * maps[idx].variable->shape[3];
      buffer = allocate_2d(buffer_size, 3);
      memset(*buffer, 0.0f, sizeof(float));
    }
  }
#endif

  char dir_visual[2][MAX_STRING_LENGTH];
  memset(dir_visual[0], 0, sizeof(dir_visual[0]));
  memset(dir_visual[1], 0, sizeof(dir_visual[1]));
  set_visual_path(dir_visual[0]);

  char dir_nn[2][MAX_STRING_LENGTH];
  memset(dir_nn[0], 0, sizeof(dir_nn[0]));
  memset(dir_nn[1], 0, sizeof(dir_nn[1]));
  set_nn_path(dir_nn[0]);

  if (create_directory(dir_visual[0]) != 0) {
    fprintf(stderr, "Can't create directory: %s\n", dir_visual[0]);
    return -1;
  }
  if (create_directory(dir_nn[0]) != 0) {
    fprintf(stderr, "Can't create directory: %s\n", dir_nn[0]);
    return -1;
  }

  strncat(dir_visual[0], "/", strlen("/"));
  strncat(dir_visual[0], run, strlen(run));
  if (create_directory(dir_visual[0]) != 0) {
    fprintf(stderr, "Can't create directory: %s\n", dir_visual[0]);
    return -1;
  }
  strncat(dir_nn[0], "/", strlen("/"));
  strncat(dir_nn[0], run, strlen(run));
  if (create_directory(dir_nn[0]) != 0) {
    fprintf(stderr, "Can't create directory: %s\n", dir_nn[0]);
    return -1;
  }

  strncat(dir_visual[0], "/", strlen("/"));
  strncat(dir_visual[0], maps[idx].out_name, strlen(maps[idx].out_name));
  if (create_directory(dir_visual[0]) != 0) {
    fprintf(stderr, "Can't create directory: %s\n", dir_visual[0]);
    return -1;
  }
  strncat(dir_nn[0], "/", strlen("/"));
  strncat(dir_nn[0], maps[idx].out_name, strlen(maps[idx].out_name));
  if (create_directory(dir_nn[0]) != 0) {
    fprintf(stderr, "Can't create directory: %s\n", dir_nn[0]);
  }

#ifdef __NVCC__
  // For the GPU implementation, we process u and v at the same time
  if (strcmp(maps[idx].name, "U") == 0) {
      set_visual_path(dir_visual[1]);
      set_path(dir_visual[1], run, maps[V].out_name);
      if (create_directory(dir_visual[1]) != 0) {
        fprintf(stderr, "Can't create directory: %s\n", dir_visual[1]);
        return -1;
      }

      set_nn_path(dir_nn[1]);
      set_path(dir_nn[1], run, maps[V].out_name);
      if (create_directory(dir_nn[1]) != 0) {
        fprintf(stderr, "Can't create directory: %s\n", dir_nn[1]);
      }
    }
#endif

#ifdef __NVCC__
  if (strcmp(maps[idx].name, "U") == 0) {
    fprintf(stdout, "----Write visual for variables: %s and %s at locs: %s %s\n", maps[idx].name,
          maps[V].name, dir_visual[0], dir_visual[1]);
    fprintf(stdout, "----Write NN inputs for variables: %s and %s at locs: %s %s\n", maps[idx].name,
          maps[V].name, dir_nn[0], dir_nn[1]);
  } else {
    fprintf(stdout, "----Write visual for variable: %s at loc: %s\n", maps[idx].name, dir_visual[0]);
    fprintf(stdout, "----Write NN input for variable: %s at loc: %s\n", maps[idx].name, dir_nn[0]);
  }
#else
  fprintf(stdout, "----Write visual for variable: %s at loc: %s\n", maps[idx].name, dir_visual[0]);
  fprintf(stdout, "----Write NN input for variable: %s at loc: %s\n", maps[idx].name, dir_nn[0]);
#endif

  uint num_layers = 0;
  if (maps[idx].variable->rank > 3 || strcmp(maps[idx].name, "ZNU") == 0 || strcmp(maps[idx].name, "ZNW") == 0) {
    num_layers = maps[idx].variable->shape[1];
    if (strcmp(maps[idx].name, "W") == 0 || strcmp(maps[idx].name, "PH") == 0 ||
        strcmp(maps[idx].name, "PHB") == 0) num_layers--;
  } else num_layers = 1;

  double i_start = cpu_second();
  for (int z = 0; z < num_layers; z++) {

    char str[4];
    convert_to_string(str, z);

    char file[2][MAX_STRING_LENGTH];
    char interpol_file[2][MAX_STRING_LENGTH];
    memset(file[0], 0, sizeof(file[0]));
    strncpy(file[0], dir_visual[0], MAX_STRING_LENGTH);
    strncat(file[0], "/", strlen("/"));
    strncat(file[0], str, strlen(str));

    char file_nn[2][MAX_STRING_LENGTH];
    memset(file_nn[0], 0, sizeof(file_nn[0]));
    strncpy(file_nn[0], dir_nn[0], MAX_STRING_LENGTH);
    strncat(file_nn[0], "/", strlen("/"));
    strncat(file_nn[0], str, strlen(str));

#ifdef __NVCC__
    if (strcmp(maps[idx].name, "U") == 0) {
      memset(file[1], 0, sizeof(file[1]));
      strcpy(file[1], dir_visual[1]);
      strncat(file[1], "/", strlen("/"));
      strncat(file[1], str, strlen(str));

      memset(file_nn[1], 0, sizeof(file_nn[1]));
      strncpy(file_nn[1], dir_nn[1], MAX_STRING_LENGTH);
      strncat(file_nn[1], "/", strlen("/"));
      strncat(file_nn[1], str, strlen(str));
    }
#endif

#ifdef __NVCC__
    if(strcmp(maps[idx].name, "U") == 0) {
#else
    if (strcmp(maps[idx].name, "U") == 0 || strcmp(maps[idx].name, "V") == 0) {
#endif
        strncpy(interpol_file[0], file[0], MAX_STRING_LENGTH);
        strncat(interpol_file[0], "_interpol", strlen("_interpol"));
        strncat(interpol_file[0], ".csv", strlen(".csv"));
#ifdef __NVCC__
        strncpy(interpol_file[1], file[1], MAX_STRING_LENGTH);
        strncat(interpol_file[1], "_interpol", strlen("_interpol"));
        strncat(interpol_file[1], ".csv", strlen(".csv"));
#endif
    }
    strncat(file[0], ".csv", strlen(".csv"));
    strncat(file_nn[0], ".csv", strlen(".csv"));
#ifdef __NVCC__
    if(strcmp(maps[idx].name, "U") == 0) {
      strncat(file[1], ".csv", strlen(".csv"));
      strncat(file_nn[1], ".csv", strlen(".csv"));
    }
#endif

    FILE *f, *f_nn;
    if (strcmp(maps[idx].name, "U") == 0 || strcmp(maps[idx].name, "V") == 0) {
      if (no_interpol_out) {
        f = fopen(file[0], "w");
      }
    } else {
      if (strcmp(maps[idx].name, "ZNU") != 0 && strcmp(maps[idx].name, "ZNW") != 0 &&
          strcmp(maps[idx].name, "W") != 0 && strcmp(maps[idx].name, "PHB") != 0 &&
          strcmp(maps[idx].name, "PH") != 0 && strcmp(maps[idx].name, "ABS_VERT_VORT") != 0 &&
          strcmp(maps[idx].name, "REL_VERT_VORT") != 0) {
            f = fopen(file[0], "w");
      }
    }

    if (strcmp(maps[idx].name, "U") != 0 && strcmp(maps[idx].name, "V") != 0 &&
        strcmp(maps[idx].name, "W") != 0 && strcmp(maps[idx].name, "PHB") != 0 &&
        strcmp(maps[idx].name, "PH") != 0 && strcmp(maps[idx].name, "ABS_VERT_VORT") != 0 &&
        strcmp(maps[idx].name, "REL_VERT_VORT") != 0) {
          f_nn = fopen(file_nn[0], "w");
    }

#ifdef __NVCC__
    FILE *f_bis;
    if(strcmp(maps[idx].name, "U") == 0) {
      if (no_interpol_out) f_bis = fopen(file[1], "w");
    }
#endif

    if (strcmp(maps[idx].name, "U") == 0 || strcmp(maps[idx].name, "V") == 0) {
      if (no_interpol_out) {
        write_visual_to_file(f, maps, idx, z);
      }
    } else {
      // Note: too early, some variables are written later!
      if (strcmp(maps[idx].name, "ZNU") != 0 && strcmp(maps[idx].name, "ZNW") != 0 &&
          strcmp(maps[idx].name, "COR_EAST") != 0 && strcmp(maps[idx].name, "COR_NORTH") != 0 &&
          strcmp(maps[idx].name, "W") != 0 && strcmp(maps[idx].name, "PHB") != 0 &&
          strcmp(maps[idx].name, "PH") != 0 && strcmp(maps[idx].name, "GEOPOTENTIAL") != 0 &&
          strcmp(maps[idx].name, "COR_PARAM") != 0 && strcmp(maps[idx].name, "ABS_VERT_VORT") != 0 &&
          strcmp(maps[idx].name, "REL_VERT_VORT") != 0) {
            write_visual_to_file(f, maps, idx, z);
      }
    }

#ifdef __NVCC__
    if (strcmp(maps[idx].name, "U") == 0) {
      if (no_interpol_out) write_visual_to_file(f_bis, maps, V, z);
    }
#endif

    // Write the scaled values for NN, here only the non-interpolated variable
    // Note: too early, some variables are written later!
    if (strcmp(maps[idx].name, "U") != 0 && strcmp(maps[idx].name, "V") != 0 &&
        strcmp(maps[idx].name, "COR_EAST") != 0 && strcmp(maps[idx].name, "COR_NORTH") != 0 &&
        strcmp(maps[idx].name, "W") != 0 && strcmp(maps[idx].name, "PHB") != 0 &&
        strcmp(maps[idx].name, "PH") != 0 && strcmp(maps[idx].name, "GEOPOTENTIAL") != 0 &&
        strcmp(maps[idx].name, "COR_PARAM") != 0 && strcmp(maps[idx].name, "ABS_VERT_VORT") != 0 &&
        strcmp(maps[idx].name, "REL_VERT_VORT") != 0) {
          write_nn_to_file(f_nn, maps, idx, z, feature_scaling_func);
    }

    if (strcmp(maps[idx].name, "U") == 0 || strcmp(maps[idx].name, "V") == 0) {
      if (no_interpol_out) {
        fclose(f);
      }
    } else {
      if (strcmp(maps[idx].name, "ZNU") != 0 && strcmp(maps[idx].name, "ZNW") != 0 &&
          strcmp(maps[idx].name, "COR_EAST") != 0 && strcmp(maps[idx].name, "COR_NORTH") != 0 &&
          strcmp(maps[idx].name, "W") != 0 && strcmp(maps[idx].name, "PHB") != 0 &&
          strcmp(maps[idx].name, "PH") != 0 && strcmp(maps[idx].name, "GEOPOTENTIAL") != 0 &&
          strcmp(maps[idx].name, "COR_PARAM") != 0 && strcmp(maps[idx].name, "ABS_VERT_VORT") != 0 &&
          strcmp(maps[idx].name, "REL_VERT_VORT") != 0) {
            fclose(f);
      }
    }

    if (strcmp(maps[idx].name, "U") != 0 && strcmp(maps[idx].name, "V") != 0 &&
        strcmp(maps[idx].name, "COR_EAST") != 0 && strcmp(maps[idx].name, "COR_NORTH") != 0 &&
        strcmp(maps[idx].name, "W") != 0 && strcmp(maps[idx].name, "PHB") != 0 &&
        strcmp(maps[idx].name, "PH") != 0 && strcmp(maps[idx].name, "GEOPOTENTIAL") != 0 &&
        strcmp(maps[idx].name, "COR_PARAM") != 0 && strcmp(maps[idx].name, "ABS_VERT_VORT") != 0 &&
        strcmp(maps[idx].name, "REL_VERT_VORT") != 0) {
          fclose(f_nn);
    }
#ifdef __NVCC__
    if(strcmp(maps[idx].name, "U") == 0) {
      if(no_interpol_out) fclose(f_bis);
    }
#endif

#ifdef __NVCC__
    int n_points = NY * NX;
    uint block_size;
    if (grid_type == STRUCTURED) {
      block_size = BLOCK_SIZE;
    } else {
      block_size = UNROLL_SIZE;
    }
    dim3 block (block_size);
    dim3 grid ((n_points + block.x-1)/block.x);
#endif

    // Interpolate the horizontal wind components at the mass points
#ifdef __NVCC__
    if (strcmp(maps[idx].name, "U") == 0) {
#else
    if (strcmp(maps[idx].name, "U") == 0 || strcmp(maps[idx].name, "V") == 0) {
#endif

#ifdef __NVCC__
      interpolate_wind_velo_horiz(maps, idx, z, interpol_file, file_nn, block, grid, grid_type, feature_scaling_func);
#else
      interpolate_wind_velo_horiz(maps, idx, z, interpol_file, file_nn, buffer, grid_type, feature_scaling_func);
#endif
    }

    if (strcmp(maps[idx].name, "W") == 0) {
#ifdef __NVCC__
      interpolate_var_vert(h_velo_w_grid, d_velo_w_grid, h_mass_grid->w, maps, idx, z, file, file_nn,
                           block, grid, grid_type, feature_scaling_func);
#else
      interpolate_var_vert(h_velo_w_grid, maps, idx, z, file, file_nn, buffer, grid_type,
                                 feature_scaling_func);
#endif
    }

    if (strcmp(maps[idx].name, "PHB") == 0) {
#ifdef __NVCC__
      interpolate_var_vert(h_base_geopot_grid, d_base_geopot_grid, h_mass_grid->phb, maps, idx, z, file,
                           file_nn, block, grid, grid_type, feature_scaling_func);
#else
      interpolate_var_vert(h_base_geopot_grid, maps, idx, z, file, file_nn, buffer, grid_type,
                           feature_scaling_func);
#endif
    }

    if (strcmp(maps[idx].name, "PH") == 0) {
#ifdef __NVCC__
      interpolate_var_vert(h_pert_geopot_grid, d_pert_geopot_grid, h_mass_grid->ph, maps, idx, z, file,
                     file_nn, block, grid, grid_type, feature_scaling_func);
#else
      interpolate_var_vert(h_pert_geopot_grid, maps, idx, z, file, file_nn, buffer, grid_type,
                     feature_scaling_func);
#endif
    }

    if (strcmp(maps[idx].name, "COR_EAST") == 0) {
      if (maps[idx].active) {
        float *lat = maps[XLAT].variable->val;
        float *coriolis = maps[COR_EAST].variable->val;

        uint nx = 0;
        uint ny = 0;

        get_horizontal_dims(COR_EAST, &nx, &ny);

        for (int y = 0; y < ny; y++) {
          for (int x = 0; x < nx; x++) {
            float v_val = coriolis[(z*(ny*nx))+((y*nx)+x)];
            float rad_lat = lat[(y*nx)+x] * M_PI/180.0f;
            float val = v_val * 2.0f * earth_angular_velocity * sinf(rad_lat);
            coriolis[(z*(ny*nx))+((y*nx)+x)] = val;
          }
        }

        write_visual_to_file(f, maps, idx, z);
        write_nn_to_file(f_nn, maps, idx, z, feature_scaling_func);
        fclose(f);
        fclose(f_nn);
      }
    }

    if (strcmp(maps[idx].name, "COR_NORTH") == 0) {
      if (maps[idx].active) {
        float *lat = maps[XLAT].variable->val;
        float *coriolis = maps[COR_NORTH].variable->val;

        uint nx = 0;
        uint ny = 0;

        get_horizontal_dims(COR_NORTH, &nx, &ny);

        for (int y = 0; y < ny; y++) {
          for (int x = 0; x < nx; x++) {
            float u_val = coriolis[(z*(ny*nx))+((y*nx)+x)];
            float rad_lat = lat[(y*nx)+x] * M_PI/180.0f;
            float val = -u_val * 2.0f * earth_angular_velocity * sinf(rad_lat);
            coriolis[(z*(ny*nx))+((y*nx)+x)] = val;
          }
        }

        write_visual_to_file(f, maps, idx, z);
        write_nn_to_file(f_nn, maps, idx, z, feature_scaling_func);
        fclose(f);
        fclose(f_nn);
      }
    }

    if (strcmp(maps[idx].name, "COR_PARAM") == 0) {
      if (maps[idx].active) {
        float *lat = maps[XLAT].variable->val;
        float *cor_param = maps[COR_PARAM].variable->val;

        uint nx = 0;
        uint ny = 0;

        get_horizontal_dims(COR_PARAM, &nx, &ny);

        for (int y = 0; y < ny; y++) {
          for (int x = 0; x < nx; x++) {
            float rad_lat = lat[(y*nx)+x] * M_PI/180.0f;
            cor_param[(z*(ny*nx))+((y*nx)+x)] = 2.0f * earth_angular_velocity * sinf(rad_lat);
          }
        }

        write_visual_to_file(f, maps, idx, z);
        write_nn_to_file(f_nn, maps, idx, z, feature_scaling_func);
        fclose(f);
        fclose(f_nn);
      }
    }

    if (strcmp(maps[idx].name, "GEOPOTENTIAL") == 0) {
      if (maps[idx].active) {
        float *phb = maps[PHB].variable->val;
        float *ph = maps[PH].variable->val;
        float *geopotential = maps[GEOPOTENTIAL].variable->val;

        uint nx = 0;
        uint ny = 0;

        get_horizontal_dims(COR_NORTH, &nx, &ny);

        for (int y = 0; y < ny; y++) {
          for (int x = 0; x < nx; x++) {
            float val = phb[(z*(ny*nx))+((y*nx)+x)] + ph[(z*(ny*nx))+((y*nx)+x)];
            geopotential[(z*(ny*nx))+((y*nx)+x)] = val;
          }
        }

        write_visual_to_file(f, maps, idx, z);
        write_nn_to_file(f_nn, maps, idx, z, feature_scaling_func);
        fclose(f);
        fclose(f_nn);
      }
    }

    if (strcmp(maps[idx].name, "ABS_VERT_VORT") == 0) {
#ifdef __NVCC__
    compute_vert_vorticity(h_abs_vert_vort, d_abs_vert_vort, maps, idx, z, file, file_nn, block, grid,
                      grid_type, feature_scaling_func, true);
#else
    // Not implemented yet
#endif
    }
    if (strcmp(maps[idx].name, "REL_VERT_VORT") == 0) {
#ifdef __NVCC__
    compute_vert_vorticity(h_rel_vert_vort, d_rel_vert_vort, maps, idx, z, file, file_nn, block, grid,
                  grid_type, feature_scaling_func, false);
#else
    // Not implemented yet
#endif
    }
  } // End z loop
  double i_elaps = cpu_second() - i_start;
  fprintf(stdout, ">>>>>>>>>>>> elapsed (%d layers): %f sec.\n",  num_layers, i_elaps);

#ifndef __NVCC__
  if (grid_type == UNSTRUCTURED) {
    if (buffer != NULL) deallocate_2d(buffer, buffer_size);
  }
#endif

  return 0;
}

int process(char files[][MAX_STRING_LENGTH], uint num_files, bool no_interpol_out, int grid_type,
            feature_scaling_pt feature_scaling_func) {

  // netcd id for the file and data variable
  int ncid;
  int ndims_in, nvars_in, ngatts_in, unlimited_in;

  // Error handling
  int retval;

  static bool first_time = false;

// ---------------------------------------------------------------
  for (int i = 0; i < num_files; i++) {
// ---------------------------------------------------------------

    char *run = NULL;
    if ((run = strstr(files[i], DATE)) == NULL) {
      fprintf(stderr, "Date not present in input netcdf file.\n");
      return -1;
    }

    // Open the file. NC_NOWRITE - read-only access to the file
    fprintf(stdout, "Processing file: %s\n", files[i]);
    if ((retval = nc_open(files[i], NC_NOWRITE, &ncid)))
      ERR(retval);

    // Retrieve some informations on variable, dimensions and global attributes
    if ((retval = nc_inq(ncid, &ndims_in, &nvars_in, &ngatts_in, &unlimited_in)))
      ERR(retval);

    fprintf(stdout, "Nb Dimension:Nb Variables:Nb attributes:id of the unlimited dimensions: %d %d %d %d\n", ndims_in, nvars_in,
        ngatts_in, unlimited_in);

    // Get a few attributes
    {
      int status;
 
      status = nc_get_att(ncid, NC_GLOBAL, CEN_LAT_NAME, &cen_lat);
      if (status != NC_NOERR) ERR(status);
      printf("%f\n", cen_lat);

      status = nc_get_att(ncid, NC_GLOBAL, CEN_LONG_NAME, &cen_long);
      if (status != NC_NOERR) ERR(status);
      printf("%f\n", cen_long);

      status = nc_get_att(ncid, NC_GLOBAL, DX_NAME, &dx);
      if (status != NC_NOERR) ERR(status);
      printf("%f\n", dx);

      status = nc_get_att(ncid, NC_GLOBAL, DY_NAME, &dy);
      if (status != NC_NOERR) ERR(status);
      printf("%f\n", dy);
    }


    if (!first_time) {
      set_maps(maps, true);
      if (check_maps(maps) != 0) {
        exit(EXIT_FAILURE);
      }

      set_maps_active(maps);
      check_depedencies(maps);
      first_time = true;
    }

    uint shape[4];
    uint rank;

    // -------------------------------------------------------------------------
    // Tensors allocation, znu, znw, the latitude and longitude (on both grids),
    // and the velocities are always allocated
    // -------------------------------------------------------------------------

    shape[0] = NT; shape[1] = NZ;
    rank = 2;
    znu = allocate_tensor(shape, rank, NULL);
    if (!maps[ZNU].active) maps[ZNU].used = true;

    shape[1] = NZ_STAG;
    znw = allocate_tensor(shape, rank, NULL);
    if (!maps[ZNW].active) maps[ZNW].used = true;

    shape[0] = NT; shape[1] = NY; shape[2] = NX;
    rank = 3;
    xlat  = allocate_tensor(shape, rank, NULL);
    xlong = allocate_tensor(shape, rank, NULL);
    if (!maps[XLAT].active) maps[XLAT].used = true;
    if (!maps[XLONG].active) maps[XLONG].used = true;

    shape[0] = NT; shape[1] = NY; shape[2] = NX_STAG;
    xlat_u  = allocate_tensor(shape, rank, NULL);
    xlong_u = allocate_tensor(shape, rank, NULL);
    if (!maps[XLAT_U].active) maps[XLAT_U].used = true;
    if (!maps[XLONG_U].active) maps[XLONG_U].used = true;

    shape[0] = NT; shape[1] = NY_STAG; shape[2] = NX;
    xlat_v  = allocate_tensor(shape, rank, NULL);
    xlong_v = allocate_tensor(shape, rank, NULL);
    if (!maps[XLAT_V].active) maps[XLAT_V].used = true;
    if (!maps[XLONG_V].active) maps[XLONG_V].used = true;

    shape[0] = NT; shape[1] = NY; shape[2] = NX;
    if (maps[SST].active) sst = allocate_tensor(shape, rank, NULL);
    if (maps[OLR].active) olr = allocate_tensor(shape, rank, NULL);

    if (maps[MU].active || maps[DRY_MASS].active) {
      mu = allocate_tensor(shape, rank, NULL);
      if (!maps[MU].active) maps[MU].used = true;
    }
    if (maps[MUB].active || maps[DRY_MASS].active) {
      mub = allocate_tensor(shape, rank, NULL);
      if (!maps[MUB].active) maps[MUB].used = true;
    }
    if (maps[DRY_MASS].active) dry_mass = allocate_tensor(shape, rank, NULL);

    shape[0] = NT; shape[1] = NZ; shape[2] = NY; shape[3] = NX;
    rank = 4;
    if (maps[CLDFRA].active)   cldfra   = allocate_tensor(shape, rank, NULL);

    if (maps[P].active || maps[PRESSURE].active) {
      p = allocate_tensor(shape, rank, NULL);
      if (!maps[P].active) maps[P].used = true;
    }
    if (maps[PB].active || maps[PRESSURE].active) {
      pb = allocate_tensor(shape, rank, NULL);
      if (!maps[PB].active) maps[PB].used = true;
    }

    bool partial = true;
    if (maps[P_HYD].active)                       phyd          = allocate_tensor(shape, rank, NULL);
    if (maps[QCLOUD].active)                      qcloud        = allocate_tensor(shape, rank, NULL);
    if (maps[QGRAUP].active)                      qgraup        = allocate_tensor(shape, rank, NULL);
    if (maps[QICE].active)                        qice          = allocate_tensor(shape, rank, NULL);
    if (maps[QNGRAUPEL].active)                   qngraupel     = allocate_tensor(shape, rank, NULL);
    if (maps[QNICE].active)                       qnice         = allocate_tensor(shape, rank, NULL);
    if (maps[QNRAIN].active)                      qnrain        = allocate_tensor(shape, rank, NULL);
    if (maps[QNSNOW].active)                      qnsnow        = allocate_tensor(shape, rank, NULL);
    if (maps[QRAIN].active)                       qrain         = allocate_tensor(shape, rank, NULL);
    if (maps[QSNOW].active)                       qsnow         = allocate_tensor(shape, rank, NULL);
    if (maps[QVAPOR].active)                      qvapor        = allocate_tensor(shape, rank, NULL);
    if (maps[T].active)                           t             = allocate_tensor(shape, rank, NULL);
    if (maps[PRESSURE].active)                    pressure      = allocate_tensor(shape, rank, NULL);
    if (maps[COR_EAST].active)                    cor_east      = allocate_tensor(shape, rank, NULL);
    if (maps[COR_NORTH].active)                   cor_north     = allocate_tensor(shape, rank, NULL);
    if (maps[GEOPOTENTIAL].active)                geopotential  = allocate_tensor(shape, rank, NULL);
    if (maps[COR_PARAM].active)                   cor_param     = allocate_tensor(shape, rank, NULL);
    if (maps[ABS_VERT_VORT].active)               abs_vert_vort = allocate_tensor(shape, rank, &partial);
    if (maps[REL_VERT_VORT].active)               rel_vert_vort = allocate_tensor(shape, rank, &partial);

    shape[1] = NZ_STAG;
    if (maps[PH].active || maps[GEOPOTENTIAL].active) {
      ph = allocate_tensor(shape, rank, NULL);
      if (!maps[PH].active) maps[PH].used = true;
    }
    if (maps[PHB].active || maps[GEOPOTENTIAL].active) {
      phb = allocate_tensor(shape, rank, NULL);
      if (!maps[PHB].active) maps[PHB].used = true;
    }

    w = allocate_tensor(shape, rank, NULL);
    if (!maps[W].active) maps[W].used = true;

    shape[1] = NZ_SOIL_STAG;
    if (maps[SH2O].active)   sh2o   = allocate_tensor(shape, rank, NULL);
    if (maps[SMCREL].active) smcrel = allocate_tensor(shape, rank, NULL);
    if (maps[SMOIS].active)  smois  = allocate_tensor(shape, rank, NULL);
    if (maps[TSLB].active)   tslb   = allocate_tensor(shape, rank, NULL);

    shape[0] = NT; shape[1] = NZ; shape[2] = NY; shape[3] = NX_STAG;
    u       = allocate_tensor(shape, rank, NULL);
    if (!maps[U].active) maps[U].used = true;

    shape[0] = NT; shape[1] = NZ; shape[2] = NY_STAG; shape[3] = NX;
    v       = allocate_tensor(shape, rank, NULL);
    if (!maps[V].active) maps[V].used = true;

    set_maps(maps, false);

    // Load the variables into memory
    for (int i = 0; i < NUM_VARIABLES; i++) {
      if ((retval = load_variable(ncid, maps[i].name, maps[i].variable, maps[i].active, maps[i].used))) {
          ERR(retval);
        }
    }

    // Compute the latitude and longitude increments
    // from the grid spacing
    {
      d_long = (dx/1000.0f) / (earth_radius * cosf(cen_lat*M_PI/180.0f));
      d_lat = (dy/1000.0f) / earth_radius;
      fprintf(stdout, "Grid spacing in degrees:\n");
      fprintf(stdout, "%f %f\n", d_long*180.0f/M_PI, d_lat*180.0f/M_PI);
    }

    // If the vorticity field is computed, set the fd tags here
    // Memory allocation using the mass points dimensions
    if (maps[ABS_VERT_VORT].active || maps[REL_VERT_VORT].active) {
      domain_tags = allocate_fd_tags(NY*NX);
      set_fd_tags(domain_tags, NY, NX);
    }

    // If required, compute the full pressure here
    if (maps[PRESSURE].active) {
      float *p = maps[P].variable->val;
      float *pb = maps[PB].variable->val;
      float *pressure = maps[PRESSURE].variable->val;

      uint nx = 0;
      uint ny = 0;

      get_horizontal_dims(PRESSURE, &nx, &ny);

      int num_layers = maps[PRESSURE].variable->shape[1];

      for (int z = 0; z < num_layers; z++) {
        for (int y = 0; y < ny; y++) {
          for (int x = 0; x < nx; x++) {
            float val = p[(z*(ny*nx))+((y*nx)+x)] + pb[(z*(ny*nx))+((y*nx)+x)];
            pressure[(z*(ny*nx))+((y*nx)+x)] = val;
          }
        }
      }
    }

    // If required, compute the full dry air mass here
    if (maps[DRY_MASS].active) {
      float *dry_mass = maps[DRY_MASS].variable->val;
      float *mu = maps[MU].variable->val;
      float *mub = maps[MUB].variable->val;

      uint nx = 0;
      uint ny = 0;

      get_horizontal_dims(DRY_MASS, &nx, &ny);

      for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
          dry_mass[(y*nx)+x] = mu[(y*nx)+x] + mub[(y*nx)+x];
        }
      }
    }

    if (grid_type == STRUCTURED) {
      // The storage for the neighbors in the x-wind and y-wind grids
      h_velo_u_grid = (velo_grid *)malloc(sizeof(velo_grid));
      h_velo_u_grid->x = (float *)malloc((NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));
      h_velo_u_grid->y = (float *)malloc((NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));
      h_velo_u_grid->val = (float *)malloc((NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));

      h_velo_v_grid = (velo_grid *)malloc(sizeof(velo_grid));
      h_velo_v_grid->x = (float *)malloc((NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));
      h_velo_v_grid->y = (float *)malloc((NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));
      h_velo_v_grid->val = (float *)malloc((NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));

      // The storage for the neighbors in the z-wind grid
      h_velo_w_grid = (velo_grid *)malloc(sizeof(velo_grid));
      h_velo_w_grid->val = (float *)malloc((NY*NX*NUM_SUPPORTING_POINTS_VERT)*sizeof(float));

      // The storage for the neighbors in the base geopotential grid
      h_base_geopot_grid = (velo_grid *)malloc(sizeof(velo_grid));
      h_base_geopot_grid->val = (float *)malloc((NY*NX*NUM_SUPPORTING_POINTS_VERT)*sizeof(float));

      // The storage for the neighbors in the perturbation geopotential grids
      h_pert_geopot_grid = (velo_grid *)malloc(sizeof(velo_grid));
      h_pert_geopot_grid->val = (float *)malloc((NY*NX*NUM_SUPPORTING_POINTS_VERT)*sizeof(float));

      double nc = cpu_second();
      // Get the neighbors coordinates of the mass points (horizontal)
      char *data_name = (char *)"coordinates";
      get_neighbors_val_horiz(h_velo_u_grid, h_velo_v_grid, maps, NY_STAG, NX_STAG, NY, NX,
        NUM_SUPPORTING_POINTS_HORIZ, data_name, NULL);
      fprintf(stdout, "Time to get neighbors coordinates in horizontal: %f sec.\n", cpu_second() - nc);
    }

#ifdef __NVCC__

    // The x-wind component memory allocs
    if (grid_type == UNSTRUCTURED) {
      h_velo_u_grid = (velo_grid *)malloc(sizeof(velo_grid));
      h_velo_u_grid->x = (float *)malloc((NY*NX_STAG)*sizeof(float));
      h_velo_u_grid->y = (float *)malloc((NY*NX_STAG)*sizeof(float));
      h_velo_u_grid->val = (float *)malloc((NZ*NY*NX_STAG)*sizeof(float));

      memcpy(h_velo_u_grid->x, xlong_u->val, (NY*NX_STAG)*sizeof(float));
      memcpy(h_velo_u_grid->y, xlat_u->val, (NY*NX_STAG)*sizeof(float));
      memcpy(h_velo_u_grid->val, u->val, (NZ*NY*NX_STAG)*sizeof(float));
   } else {
      memset(h_velo_u_grid->val, 0.0f, (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));
    }

    if(cudaMalloc((velo_grid**)&d_velo_u_grid, sizeof(velo_grid)) != cudaSuccess) {
        fprintf(stderr, "Memory allocattion failure for x-wind grid on device.\n");
        exit(EXIT_FAILURE);
    }

    if(cudaMemcpy(d_velo_u_grid, h_velo_u_grid, sizeof(velo_grid), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Memory copy <u grid> failure to device.\n");
        exit(EXIT_FAILURE);
    }

    {
      float *x[1];
      if (grid_type == STRUCTURED) {
        cudaMalloc(&(x[0]), (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));
        cudaMemcpy(&(d_velo_u_grid[0].x), &(x[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(x[0], h_velo_u_grid->x, (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float), cudaMemcpyHostToDevice);
      } else {
        cudaMalloc(&(x[0]), (NY*NX_STAG)*sizeof(float));
        cudaMemcpy(&(d_velo_u_grid[0].x), &(x[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(x[0], xlong_u->val, (NY*NX_STAG)*sizeof(float), cudaMemcpyHostToDevice);
      }
    }

    {
      float *y[1];
      if (grid_type == STRUCTURED) {
        cudaMalloc(&(y[0]), (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));
        cudaMemcpy(&(d_velo_u_grid[0].y), &(y[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(y[0], h_velo_u_grid->y, (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float), cudaMemcpyHostToDevice);
      } else {
        cudaMalloc(&(y[0]), (NY*NX_STAG)*sizeof(float));
        cudaMemcpy(&(d_velo_u_grid[0].y), &(y[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(y[0], xlat_u->val, (NY*NX_STAG)*sizeof(float), cudaMemcpyHostToDevice);
      }
    }

    {
      float *v[1];
      if (grid_type == STRUCTURED) {
        cudaMalloc(&(v[0]), (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));
        cudaMemcpy(&(d_velo_u_grid[0].val), &(v[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemset(v[0], 0.0f, (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));
      } else {
        cudaMalloc(&(v[0]), (NZ*NY*NX_STAG)*sizeof(float));
        cudaMemcpy(&(d_velo_u_grid[0].val), &(v[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(v[0], u->val, (NZ*NY*NX_STAG)*sizeof(float), cudaMemcpyHostToDevice);
      }
    }

    // The y-wind component memory allocs
    if (grid_type == UNSTRUCTURED) {
      h_velo_v_grid = (velo_grid *)malloc(sizeof(velo_grid));
      h_velo_v_grid->x = (float *)malloc((NY_STAG*NX)*sizeof(float));
      h_velo_v_grid->y = (float *)malloc((NY_STAG*NX)*sizeof(float));
      h_velo_v_grid->val = (float *)malloc((NZ*NY_STAG*NX)*sizeof(float));

      memcpy(h_velo_v_grid->x, xlong_v->val,(NY_STAG*NX)*sizeof(float));
      memcpy(h_velo_v_grid->y, xlat_v->val, (NY_STAG*NX)*sizeof(float));
      memcpy(h_velo_v_grid->val, v->val, (NZ*NY_STAG*NX)*sizeof(float));
    } else {
      memset(h_velo_v_grid->val, 0.0f, (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));
    }

    if(cudaMalloc((velo_grid**)&d_velo_v_grid, sizeof(velo_grid)) != cudaSuccess) {
        fprintf(stderr, "Memory allocation failure for y-wind grid on device.\n");
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(d_velo_v_grid, h_velo_v_grid, sizeof(velo_grid), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Memory copy <v grid> failure to device.\n");
        exit(EXIT_FAILURE);
    }

    {
      float *x[1];
      if (grid_type == STRUCTURED) {
        cudaMalloc(&(x[0]), (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));
        cudaMemcpy(&(d_velo_v_grid[0].x), &(x[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(x[0], h_velo_v_grid->x, (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float), cudaMemcpyHostToDevice);
      } else {
        cudaMalloc(&(x[0]), (NY_STAG*NX)*sizeof(float));
        cudaMemcpy(&(d_velo_v_grid[0].x), &(x[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(x[0], xlong_v->val, (NY_STAG*NX)*sizeof(float), cudaMemcpyHostToDevice);
      }
    }

    {
      float *y[1];
      if (grid_type == STRUCTURED) {
        cudaMalloc(&(y[0]), (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));
        cudaMemcpy(&(d_velo_v_grid[0].y), &(y[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(y[0], h_velo_v_grid->y, (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float), cudaMemcpyHostToDevice);
      } else {
        cudaMalloc(&(y[0]), (NY_STAG*NX)*sizeof(float));
        cudaMemcpy(&(d_velo_v_grid[0].y), &(y[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(y[0], xlat_v->val, (NY_STAG*NX)*sizeof(float), cudaMemcpyHostToDevice);
      }
    }

    {
      float *vv[1];
      if (grid_type == STRUCTURED) {
        cudaMalloc(&(vv[0]), (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));
        cudaMemcpy(&(d_velo_v_grid[0].val), &(vv[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemset(vv[0], 0.0f, (NY*NX*NUM_SUPPORTING_POINTS_HORIZ)*sizeof(float));
      } else {
        cudaMalloc(&(vv[0]), (NZ*NY_STAG*NX)*sizeof(float));
        cudaMemcpy(&(d_velo_v_grid[0].val), &(vv[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(vv[0], v->val, (NZ*NY_STAG*NX)*sizeof(float), cudaMemcpyHostToDevice);
      }
    }

    // The z-wind component memory allocs
    if (grid_type == UNSTRUCTURED) {
      fprintf(stdout, "Currently interpolation in a three dimensional unstructured grid is not fully implemented.\n");
   } else {
      memset(h_velo_w_grid->val, 0.0f, (NY*NX*NUM_SUPPORTING_POINTS_VERT)*sizeof(float));
    }

    if(cudaMalloc((velo_grid**)&d_velo_w_grid, sizeof(velo_grid)) != cudaSuccess) {
        fprintf(stderr, "Memory allocattion failure for z-wind grid on device.\n");
        exit(EXIT_FAILURE);
    }

    if(cudaMemcpy(d_velo_w_grid, h_velo_w_grid, sizeof(velo_grid), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Memory copy <w grid> failure to device.\n");
        exit(EXIT_FAILURE);
    }

    {
      float *v[1];
      if (grid_type == STRUCTURED) {
        cudaMalloc(&(v[0]), (NY*NX*NUM_SUPPORTING_POINTS_VERT)*sizeof(float));
        cudaMemcpy(&(d_velo_w_grid[0].val), &(v[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemset(v[0], 0.0f, (NY*NX*NUM_SUPPORTING_POINTS_VERT)*sizeof(float));
      } else {
        fprintf(stdout, "Currently interpolation in a three dimensional unstructured grid is not fully implemented yet.\n");
      }
    }

    // The base geopotential component memory allocs
    if (grid_type == UNSTRUCTURED) {
      fprintf(stdout, "Currently interpolation in a three dimensional unstructured grid is not fully implemented.\n");
    } else {
      memset(h_base_geopot_grid->val, 0.0f, (NY*NX*NUM_SUPPORTING_POINTS_VERT)*sizeof(float));
    }

    if(cudaMalloc((velo_grid**)&d_base_geopot_grid, sizeof(velo_grid)) != cudaSuccess) {
        fprintf(stderr, "Memory allocattion failure for base geopotential grid on device.\n");
        exit(EXIT_FAILURE);
    }

    if(cudaMemcpy(d_base_geopot_grid, h_base_geopot_grid, sizeof(velo_grid), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Memory copy <base geopotential grid> failure to device.\n");
        exit(EXIT_FAILURE);
    }

    {
      float *v[1];
      if (grid_type == STRUCTURED) {
        cudaMalloc(&(v[0]), (NY*NX*NUM_SUPPORTING_POINTS_VERT)*sizeof(float));
        cudaMemcpy(&(d_base_geopot_grid[0].val), &(v[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemset(v[0], 0.0f, (NY*NX*NUM_SUPPORTING_POINTS_VERT)*sizeof(float));
      } else {
        fprintf(stdout, "Currently interpolation in a three dimensional unstructured grid is not fully implemented yet.\n");
      }
    }

    // The perturbation geopotential component memory allocs
    if (grid_type == UNSTRUCTURED) {
      fprintf(stdout, "Currently interpolation in a three dimensional unstructured grid is not fully implemented.\n");
    } else {
      memset(h_pert_geopot_grid->val, 0.0f, (NY*NX*NUM_SUPPORTING_POINTS_VERT)*sizeof(float));
    }

    if(cudaMalloc((velo_grid**)&d_pert_geopot_grid, sizeof(velo_grid)) != cudaSuccess) {
        fprintf(stderr, "Memory allocattion failure for base geopotential grid on device.\n");
        exit(EXIT_FAILURE);
    }

    if(cudaMemcpy(d_pert_geopot_grid, h_pert_geopot_grid, sizeof(velo_grid), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Memory copy <base geopotential grid> failure to device.\n");
        exit(EXIT_FAILURE);
    }

    {
      float *v[1];
      if (grid_type == STRUCTURED) {
        cudaMalloc(&(v[0]), (NY*NX*NUM_SUPPORTING_POINTS_VERT)*sizeof(float));
        cudaMemcpy(&(d_pert_geopot_grid[0].val), &(v[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemset(v[0], 0.0f, (NY*NX*NUM_SUPPORTING_POINTS_VERT)*sizeof(float));
      } else {
        fprintf(stdout, "Currently interpolation in a three dimensional unstructured grid is not fully implemented yet.\n");
      }
    }

    // The mass grid memory allocs
    h_mass_grid = (mass_grid *)malloc(sizeof(mass_grid));
    h_mass_grid->x = (float *)malloc((NY*NX)*sizeof(float));
    h_mass_grid->y = (float *)malloc((NY*NX)*sizeof(float));
    h_mass_grid->u = (float *)malloc((NY*NX)*sizeof(float));
    h_mass_grid->v = (float *)malloc((NY*NX)*sizeof(float));
    h_mass_grid->w = (float *)malloc((NY*NX)*sizeof(float));
    h_mass_grid->ph = (float *)malloc((NY*NX)*sizeof(float));
    h_mass_grid->phb = (float *)malloc((NY*NX)*sizeof(float));

    memcpy(h_mass_grid->x, xlong->val, (NY*NX)*sizeof(float));
    memcpy(h_mass_grid->y, xlat->val, (NY*NX)*sizeof(float));
    memset(h_mass_grid->u, 0.0f, (NY*NX)*sizeof(float));
    memset(h_mass_grid->v, 0.0f, (NY*NX)*sizeof(float));
    memset(h_mass_grid->w, 0.0f, (NY*NX)*sizeof(float));
    memset(h_mass_grid->ph, 0.0f, (NY*NX)*sizeof(float));
    memset(h_mass_grid->phb, 0.0f, (NY*NX)*sizeof(float));

    if (cudaMalloc((mass_grid**)&d_mass_grid, sizeof(mass_grid)) != cudaSuccess) {
        fprintf(stderr, "Memory allocation failure for mass grid on device.\n");
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(d_mass_grid, h_mass_grid, sizeof(mass_grid), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Memory copy <mass grid> failure to device.\n");
        exit(EXIT_FAILURE);
    }
    {
      float *x[1];
      cudaMalloc(&(x[0]), (NY*NX)*sizeof(float));
      cudaMemcpy(&(d_mass_grid[0].x), &(x[0]), sizeof(float *), cudaMemcpyHostToDevice);
      cudaMemcpy(x[0], xlong->val, (NY*NX)*sizeof(float), cudaMemcpyHostToDevice);
    }
    {
      float *y[1];
      cudaMalloc(&(y[0]), (NY*NX)*sizeof(float));
      cudaMemcpy(&(d_mass_grid[0].y), &(y[0]), sizeof(float *), cudaMemcpyHostToDevice);
      cudaMemcpy(y[0], xlat->val, (NY*NX)*sizeof(float), cudaMemcpyHostToDevice);
    }
    {
      float *u[1];
      cudaMalloc(&(u[0]), (NY*NX)*sizeof(float));
      cudaMemcpy(&(d_mass_grid[0].u), &(u[0]), sizeof(float *), cudaMemcpyHostToDevice);
      cudaMemset(u[0], 0.0f, (NY*NX)*sizeof(float));
    }
    {
      float *v[1];
      cudaMalloc(&(v[0]), (NY*NX)*sizeof(float));
      cudaMemcpy(&(d_mass_grid[0].v), &(v[0]), sizeof(float *), cudaMemcpyHostToDevice);
      cudaMemset(v[0], 0.0f, (NY*NX)*sizeof(float));
    }
    {
      float *w[1];
      cudaMalloc(&(w[0]), (NY*NX)*sizeof(float));
      cudaMemcpy(&(d_mass_grid[0].w), &(w[0]), sizeof(float *), cudaMemcpyHostToDevice);
      cudaMemset(w[0], 0.0f, (NY*NX)*sizeof(float));
    }

#endif

    if (maps[ABS_VERT_VORT].active) {
      h_abs_vert_vort = (fd_container *) malloc(sizeof(fd_container));
      h_abs_vert_vort->val = (float *)malloc((NY*NX)*sizeof(float));
      h_abs_vert_vort->stencils_val = (float *)malloc((6*NY*NX)*sizeof(float));
      h_abs_vert_vort->buffer = (float *)malloc((NY*NX)*sizeof(float));
      memset(h_abs_vert_vort->val, 0.0f, (NY*NX)*sizeof(float));
      memset(h_abs_vert_vort->stencils_val, 0.0f, (6*NY*NX)*sizeof(float));
      memset(h_abs_vert_vort->buffer, 0.0f, (NY*NX)*sizeof(float));
#ifdef __NVCC__
      if (cudaMalloc((fd_container**)&d_abs_vert_vort, sizeof(fd_container)) != cudaSuccess) {
        fprintf(stderr, "Memory allocation failure for abs. vert. vorticity on device.\n");
        exit(EXIT_FAILURE);
      }

      if (cudaMemcpy(d_abs_vert_vort, h_abs_vert_vort, sizeof(fd_container), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Memory copy <abs. vert. vorticity> failure to device.\n");
        exit(EXIT_FAILURE);
      }
      {
        float *v[1];
        cudaMalloc(&(v[0]), (NY*NX)*sizeof(float));
        cudaMemcpy(&(d_abs_vert_vort[0].val), &(v[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemset(v[0], 0.0f, (NY*NX)*sizeof(float));
      }
      {
        float *v[1];
        cudaMalloc(&(v[0]), (6*NY*NX)*sizeof(float));
        cudaMemcpy(&(d_abs_vert_vort[0].stencils_val), &(v[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemset(v[0], 0.0f, (6*NY*NX)*sizeof(float));
      }
      {
        float *v[1];
        cudaMalloc(&(v[0]), (NY*NX)*sizeof(float));
        cudaMemcpy(&(d_abs_vert_vort[0].buffer), &(v[0]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemset(v[0], 0.0f, (NY*NX)*sizeof(float));
      }

      // Allocate space to store (U,V)
      maps[ABS_VERT_VORT].buffer1 = (float *)malloc((NZ*NY*NX)*sizeof(float));
      maps[ABS_VERT_VORT].buffer2 = (float *)malloc((NZ*NY*NX)*sizeof(float));

      // Copy the values of the latitude to the device
      {
        float *lat = maps[XLAT].variable->val;
        float *v[1];
        cudaMemcpy(&(v[0]), &(d_abs_vert_vort[0].buffer), sizeof(float *), cudaMemcpyDeviceToHost);
        cudaMemcpy(v[0], lat, (NY*NX)*sizeof(float), cudaMemcpyHostToDevice);
      }
#else
      fprintf(stderr, "Computing the absolute vertival vorticity is not implemented in serial yet.\n");
#endif
    }

    if (maps[REL_VERT_VORT].active) {
      h_rel_vert_vort = (fd_container *)malloc(sizeof(fd_container));
      h_rel_vert_vort->val = (float *)malloc((NY*NX)*sizeof(float));
      h_rel_vert_vort->stencils_val = (float *)malloc((6*NY*NX)*sizeof(float));
      memset(h_rel_vert_vort->val, 0.0f, (NY*NX)*sizeof(float));
      memset(h_rel_vert_vort->stencils_val, 0.0f, (6*NY*NX)*sizeof(float));
#ifdef __NVCC__
      if (cudaMalloc((fd_container**)&d_rel_vert_vort, sizeof(fd_container)) != cudaSuccess) {
        fprintf(stderr, "Memory allocation failure for rel. vert. vorticity on device.\n");
        exit(EXIT_FAILURE);
      }

      if (cudaMemcpy(d_rel_vert_vort, h_rel_vert_vort, sizeof(fd_container), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Memory copy <rel. vert. vorticity> failure to device.\n");
        exit(EXIT_FAILURE);
      }
      {
       float *v[1];
       cudaMalloc(&(v[0]), (NY*NX)*sizeof(float));
       cudaMemcpy(&(d_rel_vert_vort[0].val), &(v[0]), sizeof(float *), cudaMemcpyHostToDevice);
       cudaMemset(v[0], 0.0f, (NY*NX)*sizeof(float));
     }
     {
       float *v[1];
       cudaMalloc(&(v[0]), (6*NY*NX)*sizeof(float));
       cudaMemcpy(&(d_rel_vert_vort[0].stencils_val), &(v[0]), sizeof(float *), cudaMemcpyHostToDevice);
       cudaMemset(v[0], 0.0f, (6*NY*NX)*sizeof(float));
     }

      // Allocate space to store (U,V)
      maps[REL_VERT_VORT].buffer1 = (float *)malloc((NZ*NY*NX)*sizeof(float));
      maps[REL_VERT_VORT].buffer2 = (float *)malloc((NZ*NY*NX)*sizeof(float));
#else
      fprintf(stderr, "Computing the relative vertical vorticity is not implemented in serial yet.\n");
#endif
    }

    double i_start = cpu_second();
    for (int i = 0; i < NUM_VARIABLES; i++) {
      if (write_data(maps, i, run, no_interpol_out, grid_type, feature_scaling_func) != 0) return -1;
    }
    double i_elaps = cpu_second() - i_start;
    fprintf(stdout, ">>>>>>>>>>>> elapsed (%d variables): %f sec.\n", NUM_VARIABLES, i_elaps);

    // Close the file, freeing all ressources
    if ((retval = nc_close(ncid)))
         ERR(retval);

    deallocate_tensor(znu);
    deallocate_tensor(znw);
    deallocate_tensor(xlat);
    deallocate_tensor(xlong);
    deallocate_tensor(xlat_u);
    deallocate_tensor(xlong_u);
    deallocate_tensor(xlat_v);
    deallocate_tensor(xlong_v);

    if (sst != NULL)       deallocate_tensor(sst);
    if (mu != NULL)        deallocate_tensor(mu);
    if (mub != NULL)       deallocate_tensor(mub);
    if (dry_mass != NULL)  deallocate_tensor(dry_mass);
    if (olr != NULL)       deallocate_tensor(olr);
    if (cldfra != NULL)    deallocate_tensor(cldfra);
    if (p != NULL)         deallocate_tensor(p);
    if (pb != NULL)        deallocate_tensor(pb);
    if (ph != NULL)        deallocate_tensor(ph);
    if (phb != NULL)       deallocate_tensor(phb);
    if (phyd != NULL)      deallocate_tensor(phyd);
    if (qcloud != NULL)    deallocate_tensor(qcloud);
    if (qgraup != NULL)    deallocate_tensor(qgraup);
    if (qice != NULL)      deallocate_tensor(qice);
    if (qngraupel != NULL) deallocate_tensor(qngraupel);
    if (qnice != NULL)     deallocate_tensor(qnice);
    if (qnrain != NULL)    deallocate_tensor(qnrain);
    if (qnsnow != NULL)    deallocate_tensor(qnsnow);
    if (qrain != NULL)     deallocate_tensor(qrain);
    if (qsnow != NULL)     deallocate_tensor(qsnow);
    if (qvapor != NULL)    deallocate_tensor(qvapor);
    if (sh2o != NULL)      deallocate_tensor(sh2o);
    if (smcrel != NULL)    deallocate_tensor(smcrel);
    if (smois != NULL)     deallocate_tensor(smois);
    if (t != NULL)         deallocate_tensor(t);
    if (tslb != NULL)      deallocate_tensor(tslb);

    deallocate_tensor(u);
    deallocate_tensor(v);
    deallocate_tensor(w);

    if (pressure != NULL)      deallocate_tensor(pressure);
    if (cor_east != NULL)      deallocate_tensor(cor_east);
    if (cor_north != NULL)     deallocate_tensor(cor_north);
    if (geopotential != NULL)  deallocate_tensor(geopotential);
    if (cor_param != NULL)     deallocate_tensor(cor_param);
    if (abs_vert_vort != NULL) deallocate_tensor(abs_vert_vort);
    if (rel_vert_vort != NULL) deallocate_tensor(rel_vert_vort);

    if (domain_tags != NULL) {
      free(domain_tags);
    }

    if (grid_type == STRUCTURED) {
      free(h_velo_u_grid->x);
      free(h_velo_u_grid->y);
      free(h_velo_u_grid->val);
      free(h_velo_u_grid);

      free(h_velo_v_grid->x);
      free(h_velo_v_grid->y);
      free(h_velo_v_grid->val);
      free(h_velo_v_grid);

      free(h_velo_w_grid->val);
      free(h_velo_w_grid);

      free(h_base_geopot_grid->val);
      free(h_base_geopot_grid);

      free(h_pert_geopot_grid->val);
      free(h_pert_geopot_grid);
    }

#ifdef __NVCC__
    if (grid_type == UNSTRUCTURED) {
      free(h_velo_u_grid->x);
      free(h_velo_u_grid->y);
      free(h_velo_u_grid->val);
      free(h_velo_u_grid);

      free(h_velo_v_grid->x);
      free(h_velo_v_grid->y);
      free(h_velo_v_grid->val);
      free(h_velo_v_grid);

      free(h_velo_w_grid->val);
      free(h_velo_w_grid);

      free(h_base_geopot_grid->val);
      free(h_base_geopot_grid);

      free(h_pert_geopot_grid->val);
      free(h_pert_geopot_grid);
    }

    free(h_mass_grid->x);
    free(h_mass_grid->y);
    free(h_mass_grid->u);
    free(h_mass_grid->v);
    free(h_mass_grid->w);
    free(h_mass_grid->ph);
    free(h_mass_grid->phb);
    free(h_mass_grid);

    {
      float *x[1];
      cudaMemcpy(&(x[0]), &(d_velo_u_grid[0].x), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(x[0]);
    }
    {
      float *y[1];
      cudaMemcpy(&(y[0]), &(d_velo_u_grid[0].y), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(y[0]);
    }
    {
      float *v[1];
      cudaMemcpy(&(v[0]), &(d_velo_u_grid[0].val), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(v[0]);
    }
    cudaFree(d_velo_u_grid);

    {
      float *x[1];
      cudaMemcpy(&(x[0]), &(d_velo_v_grid[0].x), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(x[0]);
    }
    {
      float *y[1];
      cudaMemcpy(&(y[0]), &(d_velo_v_grid[0].y), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(y[0]);
    }
    {
      float *v[1];
      cudaMemcpy(&(v[0]), &(d_velo_v_grid[0].val), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(v[0]);
    }
    cudaFree(d_velo_v_grid);

    {
      float *v[1];
      cudaMemcpy(&(v[0]), &(d_velo_w_grid[0].val), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(v[0]);
    }
    cudaFree(d_velo_w_grid);

    {
      float *v[1];
      cudaMemcpy(&(v[0]), &(d_base_geopot_grid[0].val), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(v[0]);
    }
    cudaFree(d_base_geopot_grid);

    {
      float *v[1];
      cudaMemcpy(&(v[0]), &(d_pert_geopot_grid[0].val), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(v[0]);
    }
    cudaFree(d_pert_geopot_grid);

    {
      float *x[1];
      cudaMemcpy(&(x[0]), &(d_mass_grid[0].x), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(x[0]);
    }
    {
      float *y[1];
      cudaMemcpy(&(y[0]), &(d_mass_grid[0].y), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(y[0]);
    }
    {
      float *u[1];
      cudaMemcpy(&(u[0]), &(d_mass_grid[0].u), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(u[0]);
    }
    {
      float *v[1];
      cudaMemcpy(&(v[0]), &(d_mass_grid[0].v), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(v[0]);
    }
    {
      float *w[1];
      cudaMemcpy(&(w[0]), &(d_mass_grid[0].w), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(w[0]);
    }
    cudaFree(d_mass_grid);
#endif

    if (maps[ABS_VERT_VORT].active) {
      free(h_abs_vert_vort->val);
      free(h_abs_vert_vort->stencils_val);
      free(h_abs_vert_vort->buffer);
      free(h_abs_vert_vort);
#ifdef __NVCC__
      {
        float *v[1];
        cudaMemcpy(&(v[0]), &(d_abs_vert_vort[0].val), sizeof(float *), cudaMemcpyDeviceToHost);
        cudaFree(v[0]);
      }
      {
        float *v[1];
        cudaMemcpy(&(v[0]), &(d_abs_vert_vort[0].stencils_val), sizeof(float *), cudaMemcpyDeviceToHost);
        cudaFree(v[0]);
      }
      {
        float *v[1];
        cudaMemcpy(&(v[0]), &(d_abs_vert_vort[0].buffer), sizeof(float *), cudaMemcpyDeviceToHost);
        cudaFree(v[0]);
      }
      cudaFree(d_abs_vert_vort);

      free(maps[ABS_VERT_VORT].buffer1);
      free(maps[ABS_VERT_VORT].buffer2);
#endif
    }

    if (maps[REL_VERT_VORT].active) {
      free(h_rel_vert_vort->val);
      free(h_rel_vert_vort->stencils_val);
      free(h_rel_vert_vort);
#ifdef __NVCC__
      {
        float *v[1];
        cudaMemcpy(&(v[0]), &(d_rel_vert_vort[0].val), sizeof(float *), cudaMemcpyDeviceToHost);
        cudaFree(v[0]);
      }
      {
        float *v[1];
        cudaMemcpy(&(v[0]), &(d_rel_vert_vort[0].stencils_val), sizeof(float *), cudaMemcpyDeviceToHost);
        cudaFree(v[0]);
      }
      cudaFree(d_rel_vert_vort);

      free(maps[REL_VERT_VORT].buffer1);
      free(maps[REL_VERT_VORT].buffer2);
#endif
    }
// --------------------------------------------------------------------------------------------------
  } // End loop over files
// --------------------------------------------------------------------------------------------------

  return 0;
}

int main (int argc, const char *argv[]) {

  fprintf(stdout, "Program starting....\n");
  fprintf(stdout, "Increasing the program stack size to %d.\n", STACK_SIZE);

  // Increase the stack size of the program
  const rlim_t kStackSize = STACK_SIZE * 1024 * 1024;
  struct rlimit rl;

  int ret = getrlimit(RLIMIT_STACK, &rl);
  if (ret == 0) {
    if (rl.rlim_cur < kStackSize) {
      rl.rlim_cur = kStackSize;
      ret = setrlimit(RLIMIT_STACK, &rl);
      if (ret != 0) {
        fprintf(stderr, "setrlimit returned ret=%d\n", ret);
      }
    }
  }

  // Get the netcdf files to process
  char dir_files[MAX_NUMBER_FILES][MAX_STRING_LENGTH];
  char netcdf_files[MAX_NUMBER_FILES][MAX_STRING_LENGTH];
  uint num_files;
  uint num_netcdf_files = 0;

  bool no_interpol_out = false;
  if (argv[1] != NULL) {
    if (strcmp(argv[1], "-no-interpol-out") == 0) {
      no_interpol_out = true;
    }
  }

#ifdef __NVCC__
  device_info();
#endif

  get_files_from_dir(WORKDIR, dir_files, &num_files);

  char buffer[MAX_STRING_LENGTH];
  strncpy(buffer, WORKDIR, MAX_STRING_LENGTH);
  if (buffer[strlen(buffer)-1] != '/') {
    strncat(buffer, "/", strlen("/"));
  }
  for (int i = 0; i < num_files; i++) {
    if (strstr(dir_files[i], "wrfout_") != NULL) {
      strncpy(netcdf_files[num_netcdf_files], buffer, strlen(buffer));
      strncat(netcdf_files[num_netcdf_files], dir_files[i], strlen(dir_files[i]));
      num_netcdf_files++;
    }
  }

  fprintf(stdout, "Found %d files to process:\n", num_netcdf_files);
  for (int i = 0; i < num_netcdf_files; i++) {
    fprintf(stdout, "\t%s.\n", netcdf_files[i]);
  }

  // Set the feature scaling routine
  feature_scaling_pt feature_scaling_func = NULL;
  switch (FEATURE_SCALING) {
    case NORMALIZATION:
        feature_scaling_func = normalize;
    break;
    case NORMALIZATION_CENTERED:
      feature_scaling_func = normalize_centered;
    break;
    case STANDARDIZATION:
      feature_scaling_func = standardize;
    break;
  }

  double i_start = cpu_second();
  maps = allocate_maps(NUM_VARIABLES);
  if (process(netcdf_files, num_netcdf_files, no_interpol_out, GRID_TYPE, feature_scaling_func) != 0) {
    fprintf(stderr, "Program failed.\n");
  };
  free(maps);
  double i_elaps = cpu_second() - i_start;
  fprintf(stdout, ">>>>>>>>>>>> elapsed (total run): %f sec.\n", i_elaps);

  fprintf(stdout, "ALL DONE!\n");

  return 0;
}
