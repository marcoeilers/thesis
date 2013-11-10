#include <thrust/version.h>
#include <thrust/sort.h>
//#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <sys/time.h>
#include <iostream>

#define MAX_SIZE (134217728*2)
#define MIN_SIZE 8388608

int main(void)
{
  int major = THRUST_MAJOR_VERSION;
  int minor = THRUST_MINOR_VERSION;
  
//  int m = 16;

  for (int size = MIN_SIZE; size <= MAX_SIZE; size *= 2)
  {
  int *values = (int*) calloc(size, sizeof(int));
  int *result = (int*) calloc(size, sizeof(int));

  for (int i = 0; i < size; i++)
  {
    values[i] = rand();
  }

  int *dValues;
  cudaMalloc((void**) &dValues, size * sizeof(int));
 

  cudaMemcpy((void*)dValues, (void*)values, size * sizeof(int), cudaMemcpyHostToDevice);
 
  
  struct timeval before, between;

  gettimeofday(&before, NULL);
  thrust::device_ptr<int> tValues(dValues);

  
  thrust::stable_sort(tValues, tValues + size);  
  gettimeofday(&between, NULL);
  //thrust::exclusive_scan_by_key(thrust::device, tLabels, tLabels + size, tValues, tResult);

  //gettimeofday(&after, NULL);
  cudaMemcpy(result, dValues, size * sizeof(int), cudaMemcpyDeviceToHost);

  //float totalTimeNs1 = 1e9 * (after.tv_sec - between.tv_sec) + 1e3 * (after.tv_usec - between.tv_usec);
  //float nsPerInt1 = totalTimeNs1 / size;
  float totalTimeNs1 = 1e9 * (between.tv_sec - before.tv_sec) + 1e3 * (between.tv_usec - before.tv_usec);
  float nsPerInt1 = totalTimeNs1 / size;


  std::cout << size << "\t" << nsPerInt1 << std::endl;

//  for (int i = 0; i < size; i++)
//    std::cout << i << "\t" << values[i] << "\t" << result[i] << std::endl;

  cudaFree(dValues);
  free(values);
  free(result);

  }
  
  std::cout << "Thrust v" << major << "." << minor << std::endl;

  return 0;
}

