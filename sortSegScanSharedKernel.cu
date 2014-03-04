#include "cub/block/block_radix_sort.cuh"
#include "kernels/csrtools.cuh"

#define ITEMS_PER_THREAD 3
#define THREADS_PER_BLOCK 64

struct IndexValue {
	int index;
	int value;
};

struct LabelIndexValue {
	int index;
	int label;
	int value;
};



__global__ void blockSegScan(int *values, int *labels, int *result)
{
	int tid = threadIdx.x;
	typedef mgpu::CTASegScan<THREADS_PER_BLOCK, mgpu::plus<int> > SegScan;
	typedef cub::BlockRadixSort<int, THREADS_PER_BLOCK, ITEMS_PER_THREAD, IndexValue> BlockRadixSortT;

	union Shared {
		typename BlockRadixSortT::TempStorage sort;
		LabelIndexValue ilvs[THREADS_PER_BLOCK * ITEMS_PER_THREAD];
		struct {
			typename SegScan::Storage segScanStorage;
			int lastValue;
			int lastLabel;
		} segScan;
	};
	__shared__ Shared shared;
//printf("size of shared %i, sort %i, labels %i, segScanStorage %i\n", sizeof(shared), sizeof(shared.sort), sizeof(shared.labels), sizeof(shared.segScanStorage));
	int myLabels[ITEMS_PER_THREAD];
	IndexValue myValues[ITEMS_PER_THREAD];
	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		myLabels[i] = labels[ITEMS_PER_THREAD * tid + i];
		myValues[i].index = ITEMS_PER_THREAD * tid + i;
		myValues[i].value = values[ITEMS_PER_THREAD * tid + i];
	}

	BlockRadixSortT(shared.sort).Sort(myLabels, myValues);

	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		shared.ilvs[tid * ITEMS_PER_THREAD + i].index = myValues[i].index;
		shared.ilvs[tid * ITEMS_PER_THREAD + i].label = myLabels[i];
		shared.ilvs[tid * ITEMS_PER_THREAD + i].value = myValues[i].value;
	}
	__syncthreads();

	int myFlags[ITEMS_PER_THREAD];
	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		myValues[i].index = shared.ilvs[tid + i * THREADS_PER_BLOCK].index;
		myValues[i].value = shared.ilvs[tid + i * THREADS_PER_BLOCK].value;
		myLabels[i] = shared.ilvs[tid + i * THREADS_PER_BLOCK].label;
		if (i)
			myFlags[i] = shared.ilvs[tid + i * THREADS_PER_BLOCK - 1].label != shared.ilvs[tid + i * THREADS_PER_BLOCK].label;
		else
			myFlags[i] = tid ? shared.ilvs[tid + i * THREADS_PER_BLOCK - 1].label != shared.ilvs[tid + i * THREADS_PER_BLOCK].label : 1;
	}

	__syncthreads();
	int carryOut;
	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		int x = SegScan::SegScan(tid, myValues[i].value, myFlags[i], shared.segScan.segScanStorage, &carryOut, 0);

		if (i != 0 && myLabels[i] == shared.segScan.lastLabel)
		{
			x += shared.segScan.lastValue;
		}

		result[myValues[i].index] = myFlags[i] ? 0 : x;

		__syncthreads();
		if (threadIdx.x == (THREADS_PER_BLOCK - 1))
		{
			shared.segScan.lastValue = x + myValues[i].value;
			shared.segScan.lastLabel = myLabels[i];
		}
	}
}


int main()
{
	int *values = (int*) malloc(ITEMS_PER_THREAD * THREADS_PER_BLOCK * sizeof(int));
	int *flags = (int*) malloc(ITEMS_PER_THREAD * THREADS_PER_BLOCK * sizeof(int));
	int *result = (int*) malloc(ITEMS_PER_THREAD * THREADS_PER_BLOCK * sizeof(int));

	for (int i = 0; i < ITEMS_PER_THREAD * THREADS_PER_BLOCK; i++)
	{
		values[i] = i * 2;
		flags[i] = rand() % 10;
	}

	int *d_values, *d_flags, *d_result;
	cudaMalloc((void**) &d_values, ITEMS_PER_THREAD * THREADS_PER_BLOCK * sizeof(int));
	cudaMalloc((void**) &d_flags, ITEMS_PER_THREAD * THREADS_PER_BLOCK * sizeof(int));
	cudaMalloc((void**) &d_result, ITEMS_PER_THREAD * THREADS_PER_BLOCK * sizeof(int));

	cudaMemcpy(d_values, values, ITEMS_PER_THREAD * THREADS_PER_BLOCK * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_flags, flags, ITEMS_PER_THREAD * THREADS_PER_BLOCK * sizeof(int), cudaMemcpyHostToDevice);
	
	blockSegScan<<<1, THREADS_PER_BLOCK>>>(d_values, d_flags, d_result);

	cudaMemcpy(result, d_result, ITEMS_PER_THREAD * THREADS_PER_BLOCK * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < ITEMS_PER_THREAD * THREADS_PER_BLOCK; i++)
	{
		printf("index %i value %i flag %i result %i\n", i, values[i], flags[i], result[i]);
	}
}
