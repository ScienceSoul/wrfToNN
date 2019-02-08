#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <netcdf.h>

#include "interpolate.h"
#ifdef __NVCC__
  #include "gpu.h"
#endif

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

// The working directory
#define WORKDIR "/home/seddik/Documents/workdir/WRF_Jebi"

// The date of the files we are processing
#define DATE "2018"

// ----------------------------------
// The global dimensions
// ----------------------------------
uint NX      = 660;
uint NX_STAG = 661;

uint NY      = 710;
uint NY_STAG = 711;

uint NZ           = 50;
uint NZ_STAG      = 51;
uint NZ_SOIL_STAG = 4;

uint NT = 1;
// ----------------------------------

// ------------------------------------------------------
// The variables
// ------------------------------------------------------
typedef enum variables_code {
    //-----------------------------------
    // 2D variables
    //-----------------------------------
    XLAT=0,    // Latitude
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

int num_variables = 32;
// ------------------------------------------------------

// ------------------------------------------------------
// The tensors
// ------------------------------------------------------
tensor *xlat      = NULL;
tensor *xlong     = NULL;
tensor *xlat_u    = NULL;
tensor *xlong_u   = NULL;
tensor *xlat_v    = NULL;
tensor *xlong_v   = NULL;
tensor *sst       = NULL;
tensor *olr       = NULL;
tensor *cldfra    = NULL;
tensor *p         = NULL;
tensor *pb        = NULL;
tensor *phyd      = NULL;
tensor *qcloud    = NULL;
tensor *qgraup    = NULL;
tensor *qice      = NULL;
tensor *qngraupel = NULL;
tensor *qnice     = NULL;
tensor *qnrain    = NULL;
tensor *qnsnow    = NULL;
tensor *qrain     = NULL;
tensor *qsnow     = NULL;
tensor *qvapor    = NULL;
tensor *t         = NULL;
tensor *ph        = NULL;
tensor *phb       = NULL;
tensor *w         = NULL;
tensor *sh2o      = NULL;
tensor *smcrel    = NULL;
tensor *smois     = NULL;
tensor *tslb      = NULL;
tensor *u         = NULL;
tensor *v         = NULL;
// ------------------------------------------------------

// ------------------------------------------------------
// Used for CUDA if available
// ------------------------------------------------------
#ifdef __NVCC__
  velo_grid *h_velo_u_grid = NULL;
  velo_grid *h_velo_v_grid = NULL;
  mass_grid *h_mass_grid   = NULL;

  velo_grid *d_velo_u_grid = NULL;
  velo_grid *d_velo_v_grid = NULL;
  mass_grid *d_mass_grid   = NULL;
#endif
// ------------------------------------------------------

// ----------------------------------
// The variable->tensor mappings
// ----------------------------------
map *maps = NULL;
// ----------------------------------

void set_maps(map *maps) {

  for (int i = 0; i < num_variables; i++) {

    switch (i) {
      case XLAT:
        maps[i].name = "XLAT";
        maps[i].variable = xlat;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case XLAT_U:
        maps[i].name = "XLAT_U";
        maps[i].variable = xlat_u;
        maps[i].longi = xlong_u;
        maps[i].lat = xlat_u;
        break;
      case XLAT_V:
        maps[i].name = "XLAT_V";
        maps[i].variable = xlat_v;
        maps[i].longi = xlong_v;
        maps[i].lat = xlat_v;
        break;
      case XLONG:
        maps[i].name = "XLONG";
        maps[i].variable = xlong;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case XLONG_U:
        maps[i].name = "XLONG_U";
        maps[i].variable = xlong_u;
        maps[i].longi = xlong_u;
        maps[i].lat = xlat_u;
        break;
      case XLONG_V:
        maps[i].name = "XLONG_V";
        maps[i].variable = xlong_v;
        maps[i].longi = xlong_v;
        maps[i].lat = xlat_v;
        break;
      case SST:
        maps[i].name = "SST";
        maps[i].out_name = "SST";
        maps[i].variable = sst;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case OLR:
        maps[i].name = "OLR";
        maps[i].out_name = "OLR";
        maps[i].variable = olr;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case CLDFRA:
        maps[i].name = "CLDFRA";
        maps[i].out_name = "CLDFRA";
        maps[i].variable = cldfra;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case P:
        maps[i].name = "P";
        maps[i].out_name = "PERT_P";
        maps[i].variable = p;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case PB:
        maps[i].name = "PB";
        maps[i].out_name = "PB";
        maps[i].variable = pb;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case PH:
        maps[i].name = "PH";
        maps[i].out_name = "PH";
        maps[i].variable = ph;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case PHB:
        maps[i].name = "PHB";
        maps[i].out_name = "PHB";
        maps[i].variable = phb;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case P_HYD:
        maps[i].name = "P_HYD";
        maps[i].out_name = "P_HYD";
        maps[i].variable = phyd;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case QCLOUD:
        maps[i].name = "QCLOUD";
        maps[i].out_name = "QCLOUD";
        maps[i].variable = qcloud;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case QGRAUP:
        maps[i].name = "QGRAUP";
        maps[i].out_name = "QGRAUP";
        maps[i].variable = qgraup;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case QICE:
        maps[i].name = "QICE";
        maps[i].out_name = "QICE";
        maps[i].variable = qice;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case QNGRAUPEL:
        maps[i].name = "QNGRAUPEL";
        maps[i].out_name = "QNGRAUPEL";
        maps[i].variable = qngraupel;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case QNICE:
        maps[i].name = "QNICE";
        maps[i].out_name = "QNICE";
        maps[i].variable = qnice;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case QNRAIN:
        maps[i].name = "QNRAIN";
        maps[i].out_name = "QNRAIN";
        maps[i].variable = qnrain;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case QNSNOW:
        maps[i].name = "QNSNOW";
        maps[i].out_name = "QNSNOW";
        maps[i].variable = qnsnow;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case QRAIN:
        maps[i].name = "QRAIN";
        maps[i].out_name = "QRAIN";
        maps[i].variable = qrain;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case QSNOW:
        maps[i].name = "QSNOW";
        maps[i].out_name = "QSNOW";
        maps[i].variable = qsnow;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case QVAPOR:
        maps[i].name = "QVAPOR";
        maps[i].out_name = "QVAPOR";
        maps[i].variable = qvapor;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case SH2O:
        maps[i].name = "SH2O";
        maps[i].out_name = "SH2O";
        maps[i].variable = sh2o;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case SMCREL:
        maps[i].name = "SMCREL";
        maps[i].out_name = "SMCREL";
        maps[i].variable = smcrel;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case SMOIS:
        maps[i].name = "SMOIS";
        maps[i].out_name = "SMOIS";
        maps[i].variable = smois;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case T:
        maps[i].name = "T";
        maps[i].out_name = "PERT_T";
        maps[i].variable = t;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case TSLB:
        maps[i].name = "TSLB";
        maps[i].out_name = "TSLB";
        maps[i].variable = tslb;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
      case U:
        maps[i].name = "U";
        maps[i].out_name = "U";
        maps[i].variable = u;
        maps[i].mass_variable = t;
        maps[i].longi = xlong_u;
        maps[i].lat = xlat_u;
        maps[i].mass_longi = xlong;
        maps[i].mass_lat = xlat;
        break;
      case V:
        maps[i].name = "V";
        maps[i].out_name = "V";
        maps[i].variable = v;
        maps[i].mass_variable = t;
        maps[i].longi = xlong_v;
        maps[i].lat = xlat_v;
        maps[i].mass_longi = xlong;
        maps[i].mass_lat = xlat;
        break;
      case W:
        maps[i].name = "W";
        maps[i].out_name = "W";
        maps[i].variable = w;
        maps[i].longi = xlong;
        maps[i].lat = xlat;
        break;
    }
  }
}

int check_maps(map *maps) {
  int status = 0;
  for (int i = 0; i < num_variables; i++) {
    if (strcmp(maps[i].name, "XLAT")    == 0 || strcmp(maps[i].name, "XLAT_U")  == 0 ||
        strcmp(maps[i].name, "XLAT_V")  == 0 || strcmp(maps[i].name, "XLONG")   == 0 ||
        strcmp(maps[i].name, "XLONG_U") == 0 || strcmp(maps[i].name, "XLONG_V") == 0)
           continue;

    if (maps[i].name == NULL || maps[i].out_name == NULL || maps[i].variable == NULL ||
        maps[i].longi == NULL || maps[i].lat == NULL) {
      fprintf(stderr, "NULL pointer in mapping for variable %s.\n", maps[i].name);
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

int load_variable(int ncid, const char *var_name, tensor * t) {

  int retval = 0;
  int varid;

  retval = nc_inq_varid(ncid, var_name, &varid);
  retval = nc_get_var_float(ncid, varid, t->val);

  return retval;
}

void set_visual_path(char *str) {

  strcpy(str,WORKDIR);
  strncat(str, "/VISUAL", strlen("/VISUAL"));
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

void write_to_file(FILE *file, map *maps, int idx, int z) {

  uint nx = 0;
  uint ny = 0;
  if (maps[idx].variable->rank > 3) {
    ny = maps[idx].variable->shape[2];
    nx = maps[idx].variable->shape[3];
  } else {
    ny = maps[idx].variable->shape[1];
    nx = maps[idx].variable->shape[2];
  }

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

#ifdef __NVCC__
void interpolate_wind_velo(map *maps, int idx, int z, char *file, char *file_bis, dim3 block, dim3 grid) {
#else
void interpolate_wind_velo(map *maps, int idx, int z, char *file, float **buffer) {
#endif

    FILE *f = fopen(file, "w");
    fprintf(f, "longitude,latitude,%s\n", maps[idx].out_name);
#ifdef __NVCC__
    FILE *f_bis = fopen(file_bis, "w");
    fprintf(f_bis, "longitude,latitude,%s\n", maps[V].out_name);
#endif

#ifdef __NVCC__
    fprintf(stdout, "( ------ Parallel interpolate <<<%d, %d>>> %s %s at layer %d.\n", grid.x, block.x,
            maps[idx].name, maps[V].name, z);
#else
    fprintf(stdout, "( ------ Interpolate %s at layer %d\n", maps[idx].name, z);
#endif

#ifdef __NVCC__

    double i_start = cpu_second();
    radially_interpolate_gpu <<< grid, block >>>(d_velo_u_grid, d_velo_v_grid, d_mass_grid,
                             NY_STAG, NX_STAG, NY, NX, z, 2, 4, 2.0);

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
    for (int y = 0; y < maps[idx].mass_variable->shape[2]; y++) {
      for (int x = 0; x < maps[idx].mass_variable->shape[3]; x++) {
        float longi = maps[idx].mass_longi->val[(y*maps[idx].mass_longi->shape[2])+x];
        float lat   = maps[idx].mass_lat->val[(y*maps[idx].mass_lat->shape[2])+x];
        float u_val = h_mass_grid->u[(y*maps[idx].mass_variable->shape[3])+x];
        float v_val = h_mass_grid->v[(y*maps[idx].mass_variable->shape[3])+x];
        fprintf(f, "%f,%f,%f\n", longi, lat, u_val);
        fprintf(f_bis, "%f,%f,%f\n", longi, lat, v_val);
      }
    }
    i_elaps = cpu_second() -  i_start;
    fprintf(stdout, ">>>>>>>>>>>> elapsed (write file): %f sec.\n", i_elaps);

#else
    int dim = 2;
    int directions[2] = {1,2};
    bool verbose = true;

    bool reinitiate;
    if (z > 0) {
      reinitiate = false;
    } else {
      reinitiate = true;
    }

    int num_data = maps[idx].variable->shape[2] * maps[idx].variable->shape[3];
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

    for (int y = 0; y < maps[idx].mass_variable->shape[2]; y++) {
      for (int x = 0; x < maps[idx].mass_variable->shape[3]; x++) {
        float longi = maps[idx].mass_longi->val[(y*maps[idx].mass_longi->shape[2])+x];
        float lat   = maps[idx].mass_lat->val[(y*maps[idx].mass_lat->shape[2])+x];
        float val   = radially_interpolate_cpu(buffer, &longi, &lat, NULL, num_data, dim, directions,
                                               2.0f, reinitiate, 4, &verbose);
        fprintf(f, "%f,%f,%f\n", longi, lat, val);
      }
    }
#endif

    fclose(f);
#ifdef __NVCC__
    fclose(f_bis);
}
#endif

int write_visual(map *maps, int idx, char *run, bool no_interpol_out) {

  static bool first_time = true;

#ifndef __NVCC__
  float **buffer = NULL;
  uint buffer_size = 0;
#endif

  if (strcmp(maps[idx].name, "XLAT")    == 0 || strcmp(maps[idx].name, "XLAT_U")  == 0 ||
      strcmp(maps[idx].name, "XLAT_V")  == 0 || strcmp(maps[idx].name, "XLONG")   == 0 ||
      strcmp(maps[idx].name, "XLONG_U") == 0 || strcmp(maps[idx].name, "XLONG_V") == 0)
           return 0;

#ifdef __NVCC__
  if (strcmp(maps[idx].name, "V") == 0) return 0;
#endif

#ifndef __NVCC__
  if (strcmp(maps[idx].name, "U") == 0 || strcmp(maps[idx].name, "V") == 0) {
    buffer_size = maps[idx].variable->shape[2] * maps[idx].variable->shape[3];
    buffer = allocate_2d(buffer_size, 3);
    memset(*buffer, 0.0f, sizeof(float));
  }
#endif

  char dir[MAX_STRING_LENGTH];
  set_visual_path(dir);

  if (create_directory(dir) != 0) {
    fprintf(stderr, "Can't create directory: %s\n", dir);
    return -1;
  }

  strncat(dir, "/", strlen("/"));
  strncat(dir, run, strlen(run));
  if (create_directory(dir) != 0) {
    fprintf(stderr, "Can't create directory: %s\n", dir);
    return -1;
  }

  strncat(dir, "/", strlen("/"));
  strncat(dir, maps[idx].out_name, strlen(maps[idx].out_name));
  if (create_directory(dir) != 0) {
    fprintf(stderr, "Can't create directory: %s\n", dir);
    return -1;
  }

#ifdef __NVCC__
  // For the GPU implementation, we process u and v at the same time
  char dir_bis[MAX_STRING_LENGTH];
  if (strcmp(maps[idx].name, "U") == 0) {
      memset(dir_bis, 0, sizeof(dir_bis));
      set_visual_path(dir_bis);
      set_path(dir_bis, run, maps[V].out_name);
      if (create_directory(dir_bis) != 0) {
        fprintf(stderr, "Can't create directory: %s\n", dir_bis);
        return -1;
      }
    }
#endif

#ifdef __NVCC__
  if (strcmp(maps[idx].name, "U") == 0) {
    fprintf(stdout, "----Write visual for variables: %s and %s at loc: %s ", maps[idx].name,
          maps[V].name, dir);
    fprintf(stdout, "and at loc: %s\n", dir_bis);
  } else {
    fprintf(stdout, "----Write visual for variable: %s at loc: %s\n", maps[idx].name, dir);
  }
#else
  fprintf(stdout, "----Write visual for variable: %s at loc: %s\n", maps[idx].name, dir);
#endif

  uint num_layers = 0;
  if (maps[idx].variable->rank > 3) {
    num_layers = maps[idx].variable->shape[1];
  } else num_layers = 1;

  double i_start = cpu_second();
  for (int z = 0; z < num_layers; z++) {

    char str[4];
    convert_to_string(str, z);

    char file[MAX_STRING_LENGTH];
    char interpol_file[MAX_STRING_LENGTH];
    memset(file, 0, sizeof(file));
    strcpy(file, dir);
    strncat(file, "/", strlen("/"));
    strncat(file, str, strlen(str));
#ifdef __NVCC__
    char file_bis[MAX_STRING_LENGTH];
    char interpol_file_bis[MAX_STRING_LENGTH];
    if (strcmp(maps[idx].name, "U") == 0) {
      memset(file_bis, 0, sizeof(file_bis));
      strcpy(file_bis, dir_bis);
      strncat(file_bis, "/", strlen("/"));
      strncat(file_bis, str, strlen(str));
    }
#endif

#ifdef __NVCC__
    if(strcmp(maps[idx].name, "U") == 0) {
#else
    if (strcmp(maps[idx].name, "U") == 0 || strcmp(maps[idx].name, "V") == 0) {
#endif
        strcpy(interpol_file, file);
        strncat(interpol_file, "_interpol", strlen("_interpol"));
        strncat(interpol_file, ".csv", strlen(".csv"));
#ifdef __NVCC__
        strcpy(interpol_file_bis, file_bis);
        strncat(interpol_file_bis, "_interpol", strlen("_interpol"));
        strncat(interpol_file_bis, ".csv", strlen(".csv"));
#endif
    }
    strncat(file, ".csv", strlen(".csv"));
#ifdef __NVCC__
    if(strcmp(maps[idx].name, "U") == 0) {
      strncat(file_bis, ".csv", strlen(".csv"));
    }
#endif

    FILE *f;
    if (strcmp(maps[idx].name, "U") == 0 || strcmp(maps[idx].name, "V") == 0) {
      if (no_interpol_out) {
        f = fopen(file, "w");
      }
    } else {
      f = fopen(file, "w");
    }

#ifdef __NVCC__
    FILE *f_bis;
    if(strcmp(maps[idx].name, "U") == 0) {
      if (no_interpol_out) f_bis = fopen(file_bis, "w");
    }
#endif

    if (strcmp(maps[idx].name, "U") == 0 || strcmp(maps[idx].name, "V") == 0) {
      if (no_interpol_out) {
        write_to_file(f, maps, idx, z);
      }
    } else {
      write_to_file(f, maps, idx, z);
    }

#ifdef __NVCC__
    if (strcmp(maps[idx].name, "U") == 0) {
      if (no_interpol_out) write_to_file(f_bis, maps, V, z);
    }
#endif

    if (strcmp(maps[idx].name, "U") == 0 || strcmp(maps[idx].name, "V") == 0) {
      if (no_interpol_out) {
        fclose(f);
      }
    } else {
      fclose(f);
    }
#ifdef __NVCC__
    if(strcmp(maps[idx].name, "U") == 0) {
      if(no_interpol_out) fclose(f_bis);
    }
#endif

    // Interpolate the horizontal wind components at the mass points
#ifdef __NVCC__
    if (strcmp(maps[idx].name, "U") == 0) {
#else
    if (strcmp(maps[idx].name, "U") == 0 || strcmp(maps[idx].name, "V") == 0) {
#endif

#ifdef __NVCC__
    int n_points = NY * NX;
    dim3 block (UNROLL_SIZE);
    dim3 grid ((n_points + block.x-1)/block.x);
#endif

#ifdef __NVCC__
      interpolate_wind_velo(maps, idx, z, interpol_file, interpol_file_bis, block, grid);
#else
      interpolate_wind_velo(maps, idx, z, interpol_file, buffer);
#endif
    }
  } // End z loop
  double i_elaps = cpu_second() - i_start;
  fprintf(stdout, ">>>>>>>>>>>> elapsed (%d layers): %f sec.\n",  num_layers, i_elaps);

#ifndef __NVCC__
  if (buffer != NULL) deallocate_2d(buffer, buffer_size);
#endif

  return 0;
}

int process(char files[][MAX_STRING_LENGTH], uint num_files, bool no_interpol_out) {

  // netcd id for the file and data variable
  int ncid;
  int ndims_in, nvars_in, ngatts_in, unlimited_in;

  // Error handling
  int retval;

// ---------------------------------------------------------------
  for (int i = 0; i < num_files; i++) {
// ---------------------------------------------------------------

    char *run = NULL;
    if ((run = strstr(files[i], DATE)) == NULL) {
      fprintf(stderr, "Date not present in input netcdf file.\n");
      return -1;
    }

    // Open the file. NC_NOWRITE - read-only access to the file
    fprintf(stdout, "Process file: %s\n", files[i]);
    if ((retval = nc_open(files[i], NC_NOWRITE, &ncid)))
      ERR(retval);

    // Retrieve some informations on variable, dimensions and global attributes
    if ((retval = nc_inq(ncid, &ndims_in, &nvars_in, &ngatts_in, &unlimited_in)))
      ERR(retval);

    fprintf(stdout, "Nb Dimension:Nb Variables:Nb attributes:id of the unlimited dimensions: %d %d %d %d\n", ndims_in, nvars_in,
        ngatts_in, unlimited_in);

    uint shape[4];
    uint rank;

    shape[0] = NT; shape[1] = NY; shape[2] = NX;
    rank = 3;
    xlat  = allocate_tensor(shape, rank);
    xlong = allocate_tensor(shape, rank);

    shape[0] = NT; shape[1] = NY; shape[2] = NX_STAG;
    xlat_u  = allocate_tensor(shape, rank);
    xlong_u = allocate_tensor(shape, rank);

    shape[0] = NT; shape[1] = NY_STAG; shape[2] = NX;
    xlat_v  = allocate_tensor(shape, rank);
    xlong_v = allocate_tensor(shape, rank);

    shape[0] = NT; shape[1] = NY; shape[2] = NX;
    sst = allocate_tensor(shape, rank);
    olr = allocate_tensor(shape, rank);

    shape[0] = NT; shape[1] = NZ; shape[2] = NY; shape[3] = NX;
    rank = 4;
    cldfra    = allocate_tensor(shape, rank);
    p         = allocate_tensor(shape, rank);
    pb        = allocate_tensor(shape, rank);
    phyd      = allocate_tensor(shape, rank);
    qcloud    = allocate_tensor(shape, rank);
    qgraup    = allocate_tensor(shape, rank);
    qice      = allocate_tensor(shape, rank);
    qngraupel = allocate_tensor(shape, rank);
    qnice     = allocate_tensor(shape, rank);
    qnrain    = allocate_tensor(shape, rank);
    qnsnow    = allocate_tensor(shape, rank);
    qrain     = allocate_tensor(shape, rank);
    qsnow     = allocate_tensor(shape, rank);
    qvapor    = allocate_tensor(shape, rank);
    t         = allocate_tensor(shape, rank);

    shape[1] = NZ_STAG;
    ph    = allocate_tensor(shape, rank);
    phb   = allocate_tensor(shape, rank);
    w     = allocate_tensor(shape, rank);

    shape[1] = NZ_SOIL_STAG;
    sh2o   = allocate_tensor(shape, rank);
    smcrel = allocate_tensor(shape, rank);
    smois  = allocate_tensor(shape, rank);
    tslb   = allocate_tensor(shape, rank);

    shape[0] = NT; shape[1] = NZ; shape[2] = NY; shape[3] = NX_STAG;
    u       = allocate_tensor(shape, rank);

    shape[0] = NT; shape[1] = NZ; shape[2] = NY_STAG; shape[3] = NX;
    v       = allocate_tensor(shape, rank);

    set_maps(maps);
    if (check_maps(maps) != 0) {
      exit(EXIT_FAILURE);
    }


    // Load the variables into memory
    for (int i = 0; i < num_variables; i++) {
      if ((retval = load_variable(ncid, maps[i].name, maps[i].variable)))
      ERR(retval);
    }

#ifdef __NVCC__

    // The x-wind component grid
    h_velo_u_grid = (velo_grid *)malloc(sizeof(velo_grid));
    h_velo_u_grid->x = (float *)malloc((NY*NX_STAG)*sizeof(float));
    h_velo_u_grid->y = (float *)malloc((NY*NX_STAG)*sizeof(float));
    h_velo_u_grid->val = (float *)malloc((NZ*NY*NX_STAG)*sizeof(float));

    memcpy(h_velo_u_grid->x, xlong_u->val, (NY*NX_STAG)*sizeof(float));
    memcpy(h_velo_u_grid->y, xlat_u->val, (NY*NX_STAG)*sizeof(float));
    memcpy(h_velo_u_grid->val, u->val, (NZ*NY*NX_STAG)*sizeof(float));

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
      cudaMalloc(&(x[0]), (NY*NX_STAG)*sizeof(float));
      cudaMemcpy(&(d_velo_u_grid[0].x), &(x[0]), sizeof(float *), cudaMemcpyHostToDevice);
      cudaMemcpy(x[0], xlong_u->val, (NY*NX_STAG)*sizeof(float), cudaMemcpyHostToDevice);
    }
    {
      float *y[1];
      cudaMalloc(&(y[0]), (NY*NX_STAG)*sizeof(float));
      cudaMemcpy(&(d_velo_u_grid[0].y), &(y[0]), sizeof(float *), cudaMemcpyHostToDevice);
      cudaMemcpy(y[0], xlat_u->val, (NY*NX_STAG)*sizeof(float), cudaMemcpyHostToDevice);
    }
    {
      float *v[1];
      cudaMalloc(&(v[0]), (NZ*NY*NX_STAG)*sizeof(float));
      cudaMemcpy(&(d_velo_u_grid[0].val), &(v[0]), sizeof(float *), cudaMemcpyHostToDevice);
      cudaMemcpy(v[0], u->val, (NZ*NY*NX_STAG)*sizeof(float), cudaMemcpyHostToDevice);
    }

    // The y-wind component grid
    h_velo_v_grid = (velo_grid *)malloc(sizeof(velo_grid));
    h_velo_v_grid->x = (float *)malloc((NY_STAG*NX)*sizeof(float));
    h_velo_v_grid->y = (float *)malloc((NY_STAG*NX)*sizeof(float));
    h_velo_v_grid->val = (float *)malloc((NZ*NY_STAG*NX)*sizeof(float));

    memcpy(h_velo_v_grid->x, xlong_v->val,(NY_STAG*NX)*sizeof(float));
    memcpy(h_velo_v_grid->y, xlat_v->val, (NY_STAG*NX)*sizeof(float));
    memcpy(h_velo_v_grid->val, v->val, (NZ*NY_STAG*NX)*sizeof(float));

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
      cudaMalloc(&(x[0]), (NY_STAG*NX)*sizeof(float));
      cudaMemcpy(&(d_velo_v_grid[0].x), &(x[0]), sizeof(float *), cudaMemcpyHostToDevice);
      cudaMemcpy(x[0], xlong_v->val, (NY_STAG*NX)*sizeof(float), cudaMemcpyHostToDevice);
    }
    {
      float *y[1];
      cudaMalloc(&(y[0]), (NY_STAG*NX)*sizeof(float));
      cudaMemcpy(&(d_velo_v_grid[0].y), &(y[0]), sizeof(float *), cudaMemcpyHostToDevice);
      cudaMemcpy(y[0], xlat_v->val, (NY_STAG*NX)*sizeof(float), cudaMemcpyHostToDevice);
    }
    {
      float *vv[1];
      cudaMalloc(&(vv[0]), (NZ*NY_STAG*NX)*sizeof(float));
      cudaMemcpy(&(d_velo_v_grid[0].val), &(vv[0]), sizeof(float *), cudaMemcpyHostToDevice);
      cudaMemcpy(vv[0], v->val, (NZ*NY_STAG*NX)*sizeof(float), cudaMemcpyHostToDevice);
    }

    // The mass grid
    h_mass_grid = (mass_grid *)malloc(sizeof(mass_grid));
    h_mass_grid->x = (float *)malloc((NY*NX)*sizeof(float));
    h_mass_grid->y = (float *)malloc((NY*NX)*sizeof(float));
    h_mass_grid->u = (float *)malloc((NY*NX)*sizeof(float));
    h_mass_grid->v = (float *)malloc((NY*NX)*sizeof(float));

    memcpy(h_mass_grid->x, xlong->val, (NY*NX)*sizeof(float));
    memcpy(h_mass_grid->y, xlat->val, (NY*NX)*sizeof(float));
    memset(h_mass_grid->u, 0.0f, (NY*NX)*sizeof(float));
    memset(h_mass_grid->v, 0.0f, (NY*NX)*sizeof(float));

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
#endif

    double i_start = cpu_second();
    for (int i = 0; i < num_variables; i++) {
      if (write_visual(maps, i, run, no_interpol_out) != 0) return -1;
    }
    double i_elaps = cpu_second() - i_start;
    fprintf(stdout, ">>>>>>>>>>>> elapsed (%d variables): %f sec.\n",  num_variables, i_elaps);

    // Close the file, freeing all ressources
    if ((retval = nc_close(ncid)))
      ERR(retval);

    deallocate_tensor(xlat);
    deallocate_tensor(xlong);
    deallocate_tensor(xlat_u);
    deallocate_tensor(xlong_u);
    deallocate_tensor(xlat_v);
    deallocate_tensor(xlong_v);
    deallocate_tensor(sst);
    deallocate_tensor(olr);
    deallocate_tensor(cldfra);
    deallocate_tensor(p);
    deallocate_tensor(pb);
    deallocate_tensor(ph);
    deallocate_tensor(phb);
    deallocate_tensor(phyd);
    deallocate_tensor(qcloud);
    deallocate_tensor(qgraup);
    deallocate_tensor(qice);
    deallocate_tensor(qngraupel);
    deallocate_tensor(qnice);
    deallocate_tensor(qnrain);
    deallocate_tensor(qnsnow);
    deallocate_tensor(qrain);
    deallocate_tensor(qsnow);
    deallocate_tensor(qvapor);
    deallocate_tensor(sh2o);
    deallocate_tensor(smcrel);
    deallocate_tensor(smois);
    deallocate_tensor(t);
    deallocate_tensor(tslb);
    deallocate_tensor(u);
    deallocate_tensor(v);
    deallocate_tensor(w);

#ifdef __NVCC__
    free(h_velo_u_grid->x);
    free(h_velo_u_grid->y);
    free(h_velo_u_grid->val);
    free(h_velo_u_grid);

    free(h_velo_v_grid->x);
    free(h_velo_v_grid->y);
    free(h_velo_v_grid->val);
    free(h_velo_v_grid);

    free(h_mass_grid->x);
    free(h_mass_grid->y);
    free(h_mass_grid->u);
    free(h_mass_grid->v);
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
      float *vv[1];
      cudaMemcpy(&(vv[0]), &(d_velo_v_grid[0].val), sizeof(float *), cudaMemcpyDeviceToHost);
      cudaFree(vv[0]);
    }
    cudaFree(d_velo_v_grid);

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
    cudaFree(d_mass_grid);
#endif
  }
  return 0;
}

int main (int argc, const char *argv[]) {

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

  get_files_from_dir(WORKDIR, dir_files, &num_files);

  for (int i = 0; i < num_files; i++) {
    if (strstr(dir_files[i], "wrfout_") != NULL) {
      strcpy(netcdf_files[num_netcdf_files], dir_files[i]);
      num_netcdf_files++;
    }
  }

  double i_start = cpu_second();
  maps = allocate_maps(num_variables);
  if (process(netcdf_files, num_netcdf_files, no_interpol_out) != 0) {
    fprintf(stderr, "Program failed.\n");
  };
  free(maps);
  double i_elaps = cpu_second() - i_start;
  fprintf(stdout, ">>>>>>>>>>>> elapsed (total run): %f sec.\n", i_elaps);

  return 0;
}
