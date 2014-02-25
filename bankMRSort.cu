#include <stdio.h>
#include <sys/time.h>
#include <part_sort.cuh>

#define EXP 26
#define HISTS_PER_BLOCK 32
#define BUCKETS 256
#define THREADS_PER_BLOCK (BUCKETS * 4)
#define OFFSETS_THREADS_PER_BLOCK 512
#define BIN_EXP 12 // max 20
#define OFFSETS_EXP (BIN_EXP - 8)
#define SEGMENT_PREFETCH (1 << OFFSETS_EXP)

//#define DEGENERATE

__global__ void getOffsetsKernel(int *indices, int *result, int todopb, int todoTotal)
{
__shared__ int currentNums[OFFSETS_THREADS_PER_BLOCK + 1];

int start = blockIdx.x * todopb;
int *ourIndices = &indices[start];

todopb = min(todoTotal - start, todopb);

int lastIndex = start - 1;
currentNums[0] = lastIndex < 0 ? -256 : indices[lastIndex];

int i;
#pragma unroll 16
for (i = threadIdx.x; i < todopb; i += OFFSETS_THREADS_PER_BLOCK)
{
        currentNums[threadIdx.x + 1] = ourIndices[i];
        __syncthreads();

        int first = currentNums[threadIdx.x];
        int second = currentNums[threadIdx.x + 1];

        if (first / 256 != second / 256)
	{
		for (int j = (first / 256) + 1; j <= second / 256; j++)
                	result[j] = start + i;
	}
	lastIndex = currentNums[OFFSETS_THREADS_PER_BLOCK];
        __syncthreads();
	currentNums[0] = lastIndex;
}
if (start + i == todoTotal)
{
	int first = lastIndex;
	int second = (1 << BIN_EXP);

	if (first / 256 != second / 256)
        {
		
                for (int j = (first / 256) + 1; j < second / 256; j++)
		{
                        result[j] = start + i;
		}
        }

}

}


__host__ void getOffsets(int *indices, int *result, int num_elements)
{
int blocks = 8 * 16;

int todopb = num_elements / blocks;

todopb = todopb * blocks >= num_elements ? todopb : todopb + 1;

cudaFuncSetCacheConfig(getOffsetsKernel, cudaFuncCachePreferShared);

getOffsetsKernel<<<blocks, OFFSETS_THREADS_PER_BLOCK>>>(indices, result, todopb, num_elements);
}




__global__ void multiReducePassKernel(int *indices, int *values, int *offsets, int *result, int todopb, int num_elements)
{
__shared__ int blockBuckets[HISTS_PER_BLOCK * BUCKETS];
__shared__ int currentSegmentShared, currentTodoShared;

currentSegmentShared = -1;

int offset = todopb * blockIdx.x;

todopb = min(todopb, num_elements - offset);
int *blockIndices = indices + offset;
int *blockValues = values + offset;
int *myBuckets = &blockBuckets[threadIdx.x % HISTS_PER_BLOCK];



// find first segment
int currentTodo;
int currentSegment;
int prefetch = min(SEGMENT_PREFETCH, (1 << OFFSETS_EXP));

for (int i = 0; i < (1 << OFFSETS_EXP) && currentSegmentShared < 0; i += SEGMENT_PREFETCH)
{
	if (threadIdx.x < prefetch)
		blockBuckets[threadIdx.x] = offsets[i + threadIdx.x];
	__syncthreads();


	if (threadIdx.x < prefetch - 1)
	{
		int first = blockBuckets[threadIdx.x];
		int second = blockBuckets[threadIdx.x + 1];
		if  (offset >= first && offset < second)
		{
			currentSegmentShared = threadIdx.x;
			currentTodoShared = min(todopb, second - offset);
		}
	}
	__syncthreads();
}

if (currentSegmentShared < 0)
{
	currentSegment = (1 << OFFSETS_EXP) - 1;
	currentTodo = todopb;
}else{
	currentTodo = currentTodoShared;
	currentSegment = currentSegmentShared;
}

int processed = 0;


while (processed < todopb)
{
	
	// initialize buckets to zero
	for (int i = threadIdx.x; i < HISTS_PER_BLOCK * BUCKETS; i += THREADS_PER_BLOCK)
        	blockBuckets[i] = 0;

	__syncthreads();

	// try to get things aligned again
	int i;
	int currentIndex128 = min(128 - ((blockIndices - indices) % 128), currentTodo);
	for (i = threadIdx.x; i < currentIndex128; i += THREADS_PER_BLOCK)
	{
		int curInd = blockIndices[i];
                int curVal = blockValues[i];
                atomicAdd(&myBuckets[(curInd % BUCKETS) * HISTS_PER_BLOCK], curVal);
	}
	blockIndices = blockIndices + currentIndex128;
	blockValues = blockValues + currentIndex128;
	currentTodo -= currentIndex128;
	processed += currentIndex128;

	#pragma unroll 4	
	for (i = threadIdx.x; i + 3 * THREADS_PER_BLOCK < currentTodo; i += 4 * THREADS_PER_BLOCK)
	{
		int curInd = blockIndices[i];
		int curVal = blockValues[i];

	        int curInd2 = blockIndices[i + THREADS_PER_BLOCK];
	        int curVal2 = blockValues[i + THREADS_PER_BLOCK];

	        int curInd3 = blockIndices[i + 2 * THREADS_PER_BLOCK];
	        int curVal3 = blockValues[i + 2 * THREADS_PER_BLOCK];

	        int curInd4 = blockIndices[i + 3 * THREADS_PER_BLOCK];
	        int curVal4 = blockValues[i + 3 * THREADS_PER_BLOCK];

		curVal = curVal + (curInd == curInd2) * curVal2;

		atomicAdd(&myBuckets[(curInd % BUCKETS) * HISTS_PER_BLOCK], curVal);
		if (curInd != curInd2)
			atomicAdd(&myBuckets[(curInd2 % BUCKETS) * HISTS_PER_BLOCK], curVal2);

	        curVal3 = curVal3 + (curInd3 == curInd4) * curVal4;

	        atomicAdd(&myBuckets[(curInd3 % BUCKETS) * HISTS_PER_BLOCK], curVal3);
		if (curInd3 != curInd4)
		        atomicAdd(&myBuckets[(curInd4 % BUCKETS) * HISTS_PER_BLOCK], curVal4);
	}

	while (i < currentTodo)
	{
	        int curInd = blockIndices[i];
	        int curVal = blockValues[i];
	        atomicAdd(&myBuckets[(curInd % BUCKETS) * HISTS_PER_BLOCK], curVal);

		i += THREADS_PER_BLOCK;
	}
	__syncthreads();

	//copy results back to DRAM
	if (threadIdx.x < HISTS_PER_BLOCK)
	{
	int it = threadIdx.x % BUCKETS;
	int bucketOffset = BUCKETS * currentSegment;
	do {

		atomicAdd(&result[it + bucketOffset], myBuckets[it * HISTS_PER_BLOCK]);
		it = (it + 1) % BUCKETS; 
	} while(it != threadIdx.x % BUCKETS);
	}
	
	blockIndices = blockIndices + currentTodo;
	blockValues = blockValues + currentTodo;
	processed += currentTodo;
	currentSegment++;
	currentTodo = min((currentSegment  == ((1 << OFFSETS_EXP) - 1) ? num_elements : offsets[currentSegment + 1]) - offsets[currentSegment], todopb - processed);

	__syncthreads();
}

}

__host__ void multiReduce(int *indices, int *values, int *result, int num_elements)
{
// we assume that BIN_EXP > 8
cudaMemset(result, 0, (1 << BIN_EXP) * sizeof(int));

// sort by some bits

// Create a reusable sorting enactor
b40c::radix_sort::Enactor enactor;

// Create ping-pong storage wrapper
b40c::util::DoubleBuffer<int, int> double_buffer;

// The current key buffer (double_buffer.d_keys[double_buffer.selector]) backs the keys.
double_buffer.d_keys[double_buffer.selector] = indices;
double_buffer.d_values[double_buffer.selector] = values;

// Allocate pong buffer
int *d_double_indices, *d_double_values;

cudaMalloc((void**) &d_double_indices, sizeof(int) * num_elements);
cudaMalloc((void**) &d_double_values, sizeof(int) * num_elements);
double_buffer.d_keys[double_buffer.selector ^ 1] = d_double_indices;
double_buffer.d_values[double_buffer.selector ^ 1] = d_double_values;


// Sort
enactor.OneRunSort<8, OFFSETS_EXP>(double_buffer, num_elements);

cudaThreadSynchronize();
// get offsets
int *d_offsets;
cudaMalloc((void**)&d_offsets, 1 << OFFSETS_EXP * sizeof(int));

getOffsets(double_buffer.d_keys[double_buffer.selector], d_offsets, num_elements);

cudaThreadSynchronize();

//TESTING
//int *test = (int*) malloc((1 << OFFSETS_EXP) * sizeof(int));
//cudaMemcpy(test, d_offsets, (1 << OFFSETS_EXP) * sizeof(int), cudaMemcpyDeviceToHost);
//for (int i = 0; i < (1 << OFFSETS_EXP); i++)
//	printf("index %i offset %i\n", i, test[i]);

// set bank width to 32 bit
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
// prefer shared mem
cudaFuncSetCacheConfig(multiReducePassKernel, cudaFuncCachePreferShared);

cudaDeviceProp props;
cudaGetDeviceProperties(&props, 0);


int shmemPerBlock = HISTS_PER_BLOCK * BUCKETS * sizeof(int);
int blocksPerSm = props.sharedMemPerBlock / shmemPerBlock;

int totalBlocks = props.multiProcessorCount * blocksPerSm;
printf("%i SMs, %i bytes shared memory, we use %i bytes per block, therefore have %i blocks\n", props.multiProcessorCount, props.sharedMemPerBlock, shmemPerBlock, totalBlocks);

int todopb = num_elements / totalBlocks;
multiReducePassKernel<<<totalBlocks, THREADS_PER_BLOCK>>>(double_buffer.d_keys[double_buffer.selector], double_buffer.d_values[double_buffer.selector], d_offsets, result, todopb, num_elements);

}

void checkOffsets(int *keys, int *result, int num_elements)
{
	for (int i = 0; i < (1 << OFFSETS_EXP); i++)
		result[i] = 0;

	for (int i = 0; i < num_elements; i++)
		result[keys[i] / BUCKETS]++;
	int sum = 0;
	for (int i = 0; i < (1 << OFFSETS_EXP); i++)
	{
		int value = result[i];
		result[i] = sum;
		sum += value;
	}
}

void multiReduceCpu(int *keys, int *values, int *result, int num_elements)
{
	for (int i = 0; i < (1 << BIN_EXP); i++)
		result[i] = 0;

	for (int i = 0; i < num_elements; i++)
	{
		result[keys[i]] += values[i];
	}
}

int main()
{
srand(2014);
int num_elements = (1 << (EXP - 1)) * 3; 

printf("processing %i elements\n", num_elements);

int *indices, *values, *result, *result_cpu;

indices = (int*) calloc(num_elements, sizeof(int));
values = (int*) calloc(num_elements, sizeof(int));
result = (int*) calloc(1 << BIN_EXP, sizeof(int));
result_cpu = (int*) calloc(1 << BIN_EXP, sizeof(int));

#ifdef DEGENERATE
printf("Degenerate input data\n");
int index = rand() % (1 << BIN_EXP);
#endif

for (int i = 0; i < num_elements; i++)
{
#ifdef DEGENERATE
        indices[i] = index;
#else
        indices[i] = rand() % (1 << BIN_EXP);
#endif
        values[i] = rand() % 10;
}

checkOffsets(indices, result, num_elements);
for (int i = 0; i < (1 << OFFSETS_EXP); i++)
	printf("segment %i should have length %i\n", i, result[i]);

int *d_keys, *d_values, *d_result;
cudaMalloc((void**)&d_keys, num_elements * sizeof(int));
cudaMalloc((void**)&d_values, num_elements * sizeof(int));
cudaMalloc((void**)&d_result, (1 << BIN_EXP) * sizeof(int));

cudaMemcpy(d_keys, indices, num_elements * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_values, values, num_elements * sizeof(int), cudaMemcpyHostToDevice);
//cudaMemset(d_result, 0, (1 << BIN_EXP) * sizeof(int));


timeval before, after;
gettimeofday(&before, NULL);
multiReduce(d_keys, d_values, d_result, num_elements);
cudaThreadSynchronize();
gettimeofday(&after, NULL);

cudaMemcpy(result, d_result, (1 << BIN_EXP) * sizeof(int), cudaMemcpyDeviceToHost);
multiReduceCpu(indices, values, result_cpu, num_elements);

printf("%i buckets, ns/int %f\n", BUCKETS, ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / num_elements);

int correct = 1;
for (int i = 0; i < (1 << BIN_EXP); i++)
{
	if (result[i] != result_cpu[i])
	{
		correct = 0;
		printf("i sense a discrepancy! %i %i %i\n", i, result[i], result_cpu[i]);
	}
//	else
//		printf("it's all good!\n");
}

if (!correct)
	printf("result incorrect.\n");
else
	printf("result correct.\n");
}
