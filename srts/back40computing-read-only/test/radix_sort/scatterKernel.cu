#include "scatterKernel.cuh"

#include "stdio.h"

void gather(int *indices, int *values, int *result, int num_elements)
{
int threadspb = 128;
int blocks = 8 * 16;

dim3 dimBlock(threadspb, 1);
dim3 dimGrid(blocks, 1);
cudaFuncSetCacheConfig(gatherKernel, cudaFuncCachePreferL1);

gatherKernel<<<dimGrid, dimBlock>>>(indices, values, result, num_elements);
}

void gatherStoopid(int *indices, int *values, int *result, int num_elements)
{
int threadspb = 128;
int blocks = 8 * 16;

dim3 dimBlock(threadspb, 1);
dim3 dimGrid(blocks, 1);
cudaFuncSetCacheConfig(gatherStoopidKernel, cudaFuncCachePreferL1);

gatherStoopidKernel<<<dimGrid, dimBlock>>>(indices, values, result, num_elements);
}


__global__ void scatterKernel(int *indices, int *values, int *result, int num_elements)
{
//printf("in kernel\n");
	int blocks = gridDim.x;
	int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int threadspb = blockDim.x;

	int todob = num_elements / blocks;

	int todo = todob / threadspb;
	int offset = (blockId * todob) + threadId;

	int *valEnd = values + offset + (threadspb * todo);
	int *valStart = values + offset;
	int *indStart = indices + offset;

	while (valStart != valEnd)
	{
		int curVal = *valStart;
		int curInd = *indStart;
		
		result[curInd] = curVal;


		valStart += threadspb;
		indStart += threadspb;
	}
}


__global__ void multiReduceKernel(int *indices, int *values, int *result, int num_elements)
{
//printf("in kernel\n");
        int blocks = gridDim.x;
        int blockId = blockIdx.x;
        int threadId = threadIdx.x;
        int threadspb = blockDim.x;

        int todob = num_elements / blocks;

        int todo = todob / threadspb;
        int offset = (blockId * todob) + threadId;

        int *valEnd = values + offset + (threadspb * todo);
        int *valStart = values + offset;
        int *indStart = indices + offset;

        while (valStart != valEnd)
        {
                int curVal = *valStart;
                int curInd = *indStart;

                atomicAdd(&result[curInd], curVal);

                valStart += threadspb;
                indStart += threadspb;
        }
}

__global__ void histKernel(int *indices, int *result, int num_elements)
{
//printf("in kernel\n");
        int blocks = gridDim.x;
        int blockId = blockIdx.x;
        int threadId = threadIdx.x;
        int threadspb = blockDim.x;

        int todob = num_elements / blocks;

        int todo = todob / threadspb;
        int offset = (blockId * todob) + threadId;

	int *indEnd = indices + offset + (threadspb * todo);
        int *indStart = indices + offset;

        while (indStart != indEnd)
        {
                int curInd = *indStart;

                atomicAdd(&result[curInd], 1);

                indStart += threadspb;
        }
}



__global__ void gatherKernel(int *indices, int *values, int *result, int num_elements)
{
        int blocks = gridDim.x;
        int blockId = blockIdx.x;
        int threadId = threadIdx.x;
        int threadspb = blockDim.x;

        int todob = num_elements / blocks;

        int todo = todob / threadspb;
        int offset = (blockId * todob) + threadId;

        int *resEnd = result + offset + (threadspb * todo);
        int *resStart = result + offset;
        int *indStart = indices + offset;

        while (resStart != resEnd)
        {
                int curInd = *indStart;

                int curVal = values[curInd];

		*resStart = curVal;

                resStart += threadspb;
                indStart += threadspb;
        }
}


__global__ void gatherStoopidKernel(int *indices, int *values, int *result, int num_elements)
{
	int blocks = gridDim.x;
	int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int threadspb = blockDim.x;

	int todob = num_elements / blocks;

	int todo = todob / threadspb;
	int offset = (blockId * todob) + threadId * (todo);

	int *resEnd = result + offset + todo;
	int *resStart = result + offset;
	int *indStart = indices + offset;


	while (resStart != resEnd)
	{
		int curInd = *indStart;
		
		int curVal = values[curInd];

		*resStart = curVal;

		resStart++;
		indStart++;
	}
}

__global__ void scatterStoopidKernel(int *indices, int *values, int *result, int num_elements)
{
        int blocks = gridDim.x;
        int blockId = blockIdx.x;
        int threadId = threadIdx.x;
        int threadspb = blockDim.x;

        int todob = num_elements / blocks;

        int todo = todob / threadspb;
        int offset = (blockId * todob) + threadId * (todo);

        int *valEnd = values + offset + todo;
        int *valStart = values + offset;
        int *indStart = indices + offset;


        while (valStart != valEnd)
        {
                int curVal = *valStart;
                int curInd = *indStart;

                result[curInd] = curVal;

                valStart++;
                indStart++;
        }
}


void scatter(int *indices, int *values, int *result, int num_elements)
{
int threadspb = 128;
int blocks = 8 * 16;

dim3 dimBlock(threadspb, 1);
dim3 dimGrid(blocks, 1);

cudaFuncSetCacheConfig(scatterKernel, cudaFuncCachePreferL1);

scatterKernel<<<dimGrid, dimBlock>>>(indices, values, result, num_elements);
}

void multiReduce(int *indices, int *values, int *result, int num_elements)
{
int threadspb = 128;
int blocks = 8 * 16;

dim3 dimBlock(threadspb, 1);
dim3 dimGrid(blocks, 1);

cudaFuncSetCacheConfig(scatterKernel, cudaFuncCachePreferL1);

multiReduceKernel<<<dimGrid, dimBlock>>>(indices, values, result, num_elements);
}

void hist(int *indices, int *result, int num_elements)
{
int threadspb = 128;
int blocks = 8 * 16;

dim3 dimBlock(threadspb, 1);
dim3 dimGrid(blocks, 1);

cudaFuncSetCacheConfig(scatterKernel, cudaFuncCachePreferL1);

histKernel<<<dimGrid, dimBlock>>>(indices, result, num_elements);
}


void scatterStoopid(int *indices, int *values, int *result, int num_elements)
{
int threadspb = 128;
int blocks = 8 * 16;

dim3 dimBlock(threadspb, 1);
dim3 dimGrid(blocks, 1);
cudaFuncSetCacheConfig(scatterStoopidKernel, cudaFuncCachePreferL1);

scatterStoopidKernel<<<dimGrid, dimBlock>>>(indices, values, result, num_elements);
}

