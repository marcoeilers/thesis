#include "scatterKernel.cuh"
#include <assert.h>
#include <stdio.h>

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

__global__ void scatterValIndexKernel(valIndex *values, int *result, int todob)
{
__shared__ int nextValIndices[256];

        int offset = (blockIdx.x * todob);

        int *valStart = (int*) (values + offset);
	int *valEnd = valStart + todob * 2;

        while (valStart != valEnd)
        {
                nextValIndices[threadIdx.x] = valStart[threadIdx.x];
		nextValIndices[threadIdx.x + blockDim.x] = valStart[threadIdx.x + blockDim.x];
		__syncthreads();
                int curInd = nextValIndices[threadIdx.x * 2 + 1];
		int curVal = nextValIndices[threadIdx.x * 2];

                result[curInd] = curVal;

                valStart += blockDim.x * 2;
		__syncthreads();
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

//		if(curInd != 0 || curVal != 1)
//			printf("current index and value %u %i %i \n", (valStart - values), curInd, curVal);
                atomicAdd(&result[curInd], curVal);

                valStart += threadspb;
                indStart += threadspb;
        }
}

__global__ void multiReduceCombineKernel(int *indices, int *values, int *result, int num_elements, int bins, int hists)
{
//printf("in kernel\n");
        int blocks = gridDim.x;
        int blockId = blockIdx.x;
        int threadId = threadIdx.x;
        int threadspb = blockDim.x;
	int myHist =  blockIdx.x; //(hists * blockId / blocks);
        int todob = num_elements / blocks;
//if (threadIdx.x == 0) printf("block %i, my hist is %i\n", blockIdx.x, myHist);
        int todo = todob / threadspb;
        int offset = (blockId * todob) + threadId;

	int *myResult = result + myHist;

        int *valEnd = values + offset + (threadspb * todo);
        int *valStart = values + offset;
        int *indStart = indices + offset;

	if (false){
		#pragma unroll 16
                while (valStart != valEnd)
                {
                        int curVal = *valStart;
                        int curInd = *indStart;

                        myResult[curInd * hists] += curVal; 

                        valStart += threadspb;
                        indStart += threadspb;
                }

	}else{
//		#pragma unroll 16
	        while (valStart != valEnd)
	        {
	                int curVal = *valStart;
	                int curInd = *indStart;
	
	                atomicAdd(&myResult[curInd * hists], curVal);

	                valStart += threadspb;
	                indStart += threadspb;
	        }
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

void scatterValIndex(valIndex *values, int *result, int num_elements)
{
int threadspb = 128;
int blocks = num_elements / 1024;
blocks = blocks > 0 ? blocks : 1;

int todopb = num_elements / blocks;

dim3 dimBlock(threadspb, 1);
dim3 dimGrid(blocks, 1);

cudaFuncSetCacheConfig(scatterKernel, cudaFuncCachePreferL1);

scatterValIndexKernel<<<dimGrid, dimBlock>>>(values, result, todopb);
}



void multiReduce(int *indices, int *values, int *result, int num_elements)
{
int threadspb = 128;
int blocks = 8 * 16;

dim3 dimBlock(threadspb, 1);
dim3 dimGrid(blocks, 1);

cudaFuncSetCacheConfig(multiReduceKernel, cudaFuncCachePreferL1);

multiReduceKernel<<<dimGrid, dimBlock>>>(indices, values, result, num_elements);
}

void multiReduceCombine(int *indices, int *values, int *result, int num_elements, int blocks, int threadspb, int bins, int hists)
{
dim3 dimBlock(threadspb, 1);
dim3 dimGrid(blocks, 1);

cudaFuncSetCacheConfig(multiReduceCombineKernel, cudaFuncCachePreferL1);

multiReduceCombineKernel<<<dimGrid, dimBlock>>>(indices, values, result, num_elements, bins, hists);
}


void hist(int *indices, int *result, int num_elements)
{
int threadspb = 128;
int blocks = 8 * 16;

dim3 dimBlock(threadspb, 1);
dim3 dimGrid(blocks, 1);

cudaFuncSetCacheConfig(histKernel, cudaFuncCachePreferL1);

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

