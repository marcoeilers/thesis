#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
//#include <thrust/sort.h>
//#include <thrust/scatter.h>
//#include <thrust/system_error.h>
//#include <thrust/device_ptr.h>
#include "part_sort.cuh"

#define ALLOC_SIZE 1000000000

#define SIZE_NEED 1

int main() 
{
cudaError_t error;
CUdevice device;
CUcontext context;

  size_t freemem;
  size_t totalmem;

cuInit(0); // Initialize CUDA
    cuDeviceGet( &device, 0 ); // Get handle for device
    cuCtxCreate( &context, 0, device ); // Create context

  cuMemGetInfo(&freemem, &totalmem);
  printf("free %lu total %lu\n", freemem, totalmem);

int *d_zero;
error = cudaMalloc((void**) &d_zero, ALLOC_SIZE); 
if (error != cudaSuccess)
    printf("error during initial cudamalloc\n");

int size = 1 << 20;

  while (size * SIZE_NEED * sizeof(int) < ALLOC_SIZE)
    {
      
      int *keys = (int*) malloc(size * sizeof(int));
      int *values = (int*) malloc(size * sizeof(int));
//      int *result = (int*) malloc(size * sizeof(int));

      for (int i = 0; i < size; i++)
	{
	  keys[i] = rand() % size;
	  values[i] = rand();
	}
      
      
      
//      int *d_keys, *d_values, *d_result;
/*
      if (cudaSuccess != cudaMalloc((void**) &d_keys, size * sizeof(int)))
        printf("error cudaMalloc %i", cudaGetLastError());
      if (cudaSuccess != cudaMalloc((void**) &d_values, size * sizeof(int)))
        printf("error cudaMalloc");
      if (cudaSuccess != cudaMalloc((void**) &d_result, size * sizeof(int)))
	printf("error cudaMalloc");
*/

//      d_keys = d_zero;
//      d_values = d_keys + size;
//      d_result = d_values + size;

//      cudaThreadSynchronize(); 
//      thrust::device_ptr<int> keys_ptr(d_keys );
//      thrust::device_ptr<int> values_ptr(d_values );
//      thrust::device_ptr<int> result_ptr(d_result);

//      timeval before, after;
//      gettimeofday(&before, NULL);
/*
      error = cudaMemcpy(d_values, values, sizeof(int) * size, cudaMemcpyHostToDevice);
      if (error != cudaSuccess)
          printf("error copy values\n");

      error = cudaMemcpy(d_keys, keys, sizeof(int) * size, cudaMemcpyHostToDevice);
      if (error != cudaSuccess)
          printf("error during copy keys\n");
*/     
//      thrust::stable_sort_by_key(keys_ptr, keys_ptr + size, values_ptr);

//      thrust::scatter(values_ptr, values_ptr + size, keys_ptr, result_ptr);
//      thrust::gather(keys_ptr, keys_ptr + size, values_ptr, result_ptr);
      timeval result;
      result = one_run_sort<0, 16>(keys, values, size, false);

//      cudaThreadSynchronize();
//      gettimeofday(&after, NULL);

      float time;
      time = result.tv_sec * 1e9 + result.tv_usec * 1e3;
     
      printf("%i\t%f\n", size, time / size);
      
    
      free(keys);
      free(values);
//      free(result);
      size *= 2;
    }
cudaFree(&d_zero);
}


