#ifndef SCATTERKERNEL
#define SCATTERKERNEL

#include "valIndex.h"

void scatter(int *indices, int *values, int *result, int num_elements);
void scatterValIndex(valIndex *values, int *result, int num_elements);
void multiReduce(int *indices, int *values, int *result, int num_elements);
void hist(int *indices, int *result, int num_elements);
void scatterStoopid(int *indices, int *values, int *result, int num_elements);
void gather(int *indices, int *values, int *result, int num_elements);
void gatherStoopid(int *indices, int *values, int *result, int num_elements);

__global__ void scatterValIndexKernel(valIndex *values, int *result, int num_elements);
__global__ void scatterKernel(int *indices, int *values, int *result, int num_elements);
__global__ void scatterStoopidKernel(int *indices, int *values, int *result, int num_elements);
__global__ void gatherKernel(int *indices, int *values, int *result, int num_elements);
__global__ void gatherStoopidKernel(int *indices, int *values, int *result, int num_elements);

#endif
