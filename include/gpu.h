# ifndef gpu_h
#define gpu_h

#include <sys/time.h>
#include <cuda_runtime.h>

#define CHECK(call) {                                                    \
	const cudaError_t error = call;                                        \
	if (error != cudaSuccess) {                                            \
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);               \
		fprintf(stderr, "code:%d, reason:%s.\n", cudaGetErrorString(error)); \
		exit(-1);                                                            \
	}                                                                      \
}

void device_info(void);
double cpu_second(void);

#endif
