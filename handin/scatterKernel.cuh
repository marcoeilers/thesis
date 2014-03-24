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
void multiReduceCombine(int *indices, int *values, int *result, int num_elements, int blocks, int threadspb, int bins, int hists);


template <int NT, int VT>
__global__ void multiReduceCombineTransposeKernel(int *indices, int *values, int *result, int num_elements, int segLength)
{
// assume NT >= 32 and multiple of 32
// assume VT < 32
        __shared__ int sIndices[VT * NT];
        __shared__ int sValues[VT * NT];
        int blocks = gridDim.x;
        int blockId = blockIdx.x;
        int threadId = threadIdx.x;
//        int threadspb = blockDim.x;
        int hists = blocks * NT;
        int *myBuckets =  result + blockId * NT + threadId;
        int todopb = NT * segLength;

        int *ourIndices = indices + blockId * todopb;
        int *ourValues = values + blockId * todopb;

        int *myIndices = &sIndices[threadId * VT];
        int *myValues = &sValues[threadId * VT];

//        int warps = threadspb / 32;
//        int laneId = threadIdx.x % 32;
//        int warpId = threadIdx.x / 32;

        #pragma unroll 16
        for (int it = 0; it < segLength; it += VT)
        {
        int offset = it;
        int todo = min(segLength - offset, VT);

	for(int i = threadIdx.x; i < todo * NT; i += NT)
	{
		int segm = i / todo;
		int ind = i % todo;
		int sIndex = segm * VT + ind;
		int index = segm * segLength + ind + offset;
		sIndices[sIndex] = ourIndices[index];
		sValues[sIndex] = ourValues[index];
	}
/*
        if (laneId < todo)
        {
		int myOffset = offset + laneId;
                for (int i = warpId; i < NT; i += warps)
                {
                        sIndices[i * VT + laneId] = ourIndices[i * segLength + myOffset];
                        sValues[i * VT + laneId] = ourValues[i * segLength + myOffset];
                }
        }
*/
        __syncthreads();

        #pragma unroll 31
        for (int i = 0; i < todo; i++)
        {
//		myBuckets[hists * sIndices[i + threadId * VT]] += sValues[i + threadId * VT];	
                myBuckets[hists * myIndices[i]] += myValues[i];
//		atomicAdd(&myBuckets[hists * myIndices[i]], myValues[i]);
        }
        __syncthreads();
        }
}

template<int NT, int VT>
void multiReduceCombineTranspose(int *indices, int *values, int *result, int num_elements, int blocks, int segLength)
{
dim3 dimBlock(NT, 1);
dim3 dimGrid(blocks, 1);

cudaFuncSetCacheConfig(multiReduceCombineTransposeKernel<NT, VT>, cudaFuncCachePreferShared);

multiReduceCombineTransposeKernel<NT, VT><<<dimGrid, dimBlock>>>(indices, values, result, num_elements, segLength);
}


__global__ void scatterValIndexKernel(valIndex *values, int *result, int num_elements);
__global__ void scatterKernel(int *indices, int *values, int *result, int num_elements);
__global__ void scatterStoopidKernel(int *indices, int *values, int *result, int num_elements);
__global__ void gatherKernel(int *indices, int *values, int *result, int num_elements);
__global__ void gatherStoopidKernel(int *indices, int *values, int *result, int num_elements);

#endif
