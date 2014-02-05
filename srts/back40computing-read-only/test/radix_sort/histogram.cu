#include <part_sort.cuh>
#include <sys/time.h>
#include <scatterKernel.cuh>
           

#define BIN_EXP 8
#define BITS 0
#define EXP 26
#define START (BIN_EXP - BITS)

#define CHECK_RESULTS


void cpuHist(int *h_keys, int *h_result, int num_elements)
{
for (int i = 0; i < num_elements; i++)
{
int curInd = h_keys[i];
h_result[curInd]++;
}
}  

void compare(int *result1, int *result2, int num_elements)
{
for (int i = 0; i < num_elements; i++)
{
if (result1[i] != result2[i])
	printf("it's just not the same.\n");
}
}

template<int START_BIT, int NO_BITS>
std::pair<timeval, timeval> sortHist(int *h_keys, int *h_result, int num_elements, bool keys_only)
{
	typedef int KeyType;
	typedef int ValueType;
	
	// Allocate device data. (We will let the sorting enactor create
	// the "pong" storage if/when necessary.)
	KeyType *d_keys;
	int *d_result;
	cudaMalloc((void**) &d_keys, sizeof(KeyType) * num_elements);

	// Copy host data to device data
	cudaMemcpy(d_keys, h_keys, sizeof(KeyType) * num_elements, cudaMemcpyHostToDevice);

	timeval before, between, after;
	
	gettimeofday(&before, NULL);



//++++++ doSortDevice

	// Create a reusable sorting enactor
	b40c::radix_sort::Enactor enactor;

	// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<KeyType> double_buffer;

		// The current key buffer (double_buffer.d_keys[double_buffer.selector]) backs the keys.
		double_buffer.d_keys[double_buffer.selector] = d_keys;

		// Allocate pong buffer
		cudaMalloc((void**) &double_buffer.d_keys[double_buffer.selector ^ 1], sizeof(KeyType) * num_elements);

		// Sort
//		enactor.Sort(double_buffer, num_elements);
		enactor.OneRunSort<START_BIT, NO_BITS>(double_buffer, num_elements);


	cudaThreadSynchronize();
//printf("error after sort %i\n", cudaGetLastError());

        gettimeofday(&between, NULL);

        cudaMalloc((void**) &d_result, sizeof(int) * num_elements);

#ifdef CHECK_RESULTS
	cudaMemset((void*) d_result, 0, sizeof(int) * (1 << BIN_EXP));
#endif
//printf("error after malloc %i\n", cudaGetLastError());
        hist(d_keys, d_result, num_elements);
//printf("error after scatter %i\n", cudaGetLastError());
        cudaThreadSynchronize();
        gettimeofday(&after, NULL);

		// Cleanup "pong" storage
		if (double_buffer.d_keys[double_buffer.selector ^ 1]) {
			cudaFree(double_buffer.d_keys[double_buffer.selector ^ 1]);
		}


//------ doSortDevice
/*


	int *valuesPtr = doSortDevice<START_BIT, NO_BITS>(d_keys, d_values, num_elements, keys_only);
	cudaThreadSynchronize();
printf("error after sort %i\n", cudaGetLastError());

	gettimeofday(&between, NULL);

	cudaMalloc((void**) &d_result, sizeof(int) * (1 << BINS_EXP));
printf("error after malloc %i\n", cudaGetLastError());
	scatter(d_keys, valuesPtr, d_result, num_elements);
printf("error after scatter %i\n", cudaGetLastError());
	cudaThreadSynchronize();
//int error = cudaGetLastError();
//printf("error after scatter %i\n", error);
	gettimeofday(&after, NULL);
*/
	cudaMemcpy(h_result, d_result, sizeof(int) * (1 << BIN_EXP), cudaMemcpyDeviceToHost);


	cudaFree(d_keys);

	timeval result1, result2;
	result2.tv_sec = after.tv_sec - between.tv_sec;
	result2.tv_usec = after.tv_usec - between.tv_usec;
	result1.tv_sec = between.tv_sec - before.tv_sec;
	result1.tv_usec = between.tv_usec - before.tv_usec;
	std::pair<timeval, timeval> result(result1, result2);
	return result;
}

int selfsimilar(int N, double h) 
{
return (1 + (int) 
	(N * 
		pow( (rand() / RAND_MAX), log(h)/log(1.0-h) ) ));
}

int main()
{
	printf("rand max is %i\n", RAND_MAX);
	typedef int KeyType;
	typedef int ValueType;

	unsigned int num_elements = 1 << EXP;


        // Allocate host problem data
	KeyType *h_keys = new KeyType[num_elements];
	int *h_result = new int[1<<BIN_EXP];
	int *ref_result = new int[1 << BIN_EXP];
	int *indices = new int[1 << BIN_EXP];

	int index = rand() % (1 << BIN_EXP);
	for (int i = 0; i < (1 << BIN_EXP); i++)
	{
		indices[i] = rand() % (1 << BIN_EXP);
	}
        // Initialize host problem data

        for (int i = 0; i < num_elements; ++i)
        {
              h_keys[i] = rand() % (1 << BIN_EXP);
//		h_keys[i] = indices[selfsimilar(1 << BIN_EXP, 0.2)];
//		h_keys[i] = index;
       }

	for (int i = 0; i < (1 << BIN_EXP); i++)
	{
		ref_result[i] = 0;
	}
	std::pair<timeval, timeval> result;
	result = sortHist<START, BITS>(h_keys, h_result, num_elements, false);

#ifdef CHECK_RESULTS
	cpuHist(h_keys, ref_result, num_elements);
	compare(h_result, ref_result, 1 << BIN_EXP);
#endif

	float time1 = result.first.tv_sec * 1e9 + result.first.tv_usec * 1e3;
	float time2 = result.second.tv_sec * 1e9 + result.second.tv_usec * 1e3;
	printf("%i\t%i\t%f\t%f\t%f\n", num_elements, BITS, time1 / num_elements, time2 / num_elements, (time1 + time2) / num_elements);

	delete(h_keys);
	delete(h_result);
	return 0;
}
