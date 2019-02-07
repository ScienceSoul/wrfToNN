#include <stdio.h>
#include "gpu.h"

void device_info(void) {

  int device_count = 0;
  CHECK(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stdout, "There are no available device(s) that suport CUDA.\n");
  } else {
    fprintf(stdout, "Detected %d CUDA capable device(s).\n", device_count);
  }

  int dev, driver_version = 0, runtime_version = 0;

  dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, dev);
  fprintf(stdout, "Device %d: \"%s\".\n", dev, device_prop.name);

  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);
  fprintf(stdout, "   CUDA Driver Version / Runtime Version   %d.%d / %d.%d.\n",
                  driver_version/1000, (driver_version%100)/10,
                  runtime_version/1000, (runtime_version%100)/10);
  fprintf(stdout, "   CUDA Capability Major / Minor version number:   %d.%d.\n",
                  device_prop.major, device_prop.minor);
  fprintf(stdout, "   Number of multiprocessors:    %d.\n",
                  device_prop.multiProcessorCount);
  fprintf(stdout, "   Total amount of global memory:    %.2f GBytes (%llu bytes).\n",
                  (float)device_prop.totalGlobalMem/(powf(1024.0,3)),
                  (unsigned long long)device_prop.totalGlobalMem);
  fprintf(stdout, "   GPU Clock rate:   %.0f Mhz (%0.2f Ghz).\n",
                   device_prop.clockRate * 1.0e-3f,
                   device_prop.clockRate * 1.0e-6f);
  fprintf(stdout, "   Memory Clock rate:    %.0f Mhz.\n",
                   device_prop.memoryClockRate * 1.0e-3f);
  fprintf(stdout, "   Memory Bus width:     %d-bit.\n",
                  device_prop.memoryBusWidth);
  if (device_prop.l2CacheSize) {
    fprintf(stdout, "   L2 Cache size:    %d bytes.\n", device_prop.l2CacheSize);
  }
  fprintf(stdout, "   Max Texture Dimension Size (x, y, z): 1D=(%d), 2D=(%d,%d),3D=(%d,%d,%d).\n",
                   device_prop.maxTexture1D,
                   device_prop.maxTexture2D[0], device_prop.maxTexture2D[1],
                   device_prop.maxTexture3D[0], device_prop.maxTexture3D[1],
                   device_prop.maxTexture3D[2]);
  fprintf(stdout, "   Max Layered Texture Size (dim) x layers: 1D=(%d) x %d, 2D=(%d,%d) x %d.\n",
                   device_prop.maxTexture1DLayered[0], device_prop.maxTexture1DLayered[1],
                   device_prop.maxTexture2DLayered[0], device_prop.maxTexture2DLayered[1],
                   device_prop.maxTexture2DLayered[2]);
  fprintf(stdout, "   Total amount of constant memory:    %lu bytes.\n",
                   device_prop.totalConstMem);
  fprintf(stdout, "   Total amount of shared memory per block:    %lu bytes.\n",
                  device_prop.sharedMemPerBlock);
  fprintf(stdout, "   Total number of registers available per block:    %d.\n",
                  device_prop.regsPerBlock);
  fprintf(stdout, "   Warp size:    %d\n", device_prop.warpSize);
  fprintf(stdout, "   Maximum number of threads per multiprocessor:   %d.\n",
                   device_prop.maxThreadsPerMultiProcessor);
  fprintf(stdout, "   Maximum number of threads per block:    %d.\n",
                   device_prop.maxThreadsPerBlock);
  fprintf(stdout, "   Maximum number of warps per multiprocessor:   %d.\n",
                   device_prop.maxThreadsPerMultiProcessor/32);
  fprintf(stdout, "   Maximum sizes of each dimension of a block:   %d x %d x %d.\n",
                   device_prop.maxGridSize[0], device_prop.maxGridSize[1],
                   device_prop.maxGridSize[2]);
  fprintf(stdout, "   Maximum memory pitch:   %lu bytes.\n",
                    device_prop.memPitch);
}

double cpu_second(void) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.0e-6);
}
