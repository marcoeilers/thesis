#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

struct spinerec {
        int rowsum;
        int spinesum;
        spinerec * spine;
};

__global__
void initialize(spinerec *td, spinerec *bd, int *labels)
{

        int i = threadIdx.x + (blockIdx.x * blockDim.x);

        td[i].rowsum = 0;

        td[i].spinesum = 0;

        td[i].spine = &(bd[labels[i]]);

        bd[labels[i]].spine = &bd[labels[i]];
        bd[labels[i]].rowsum = 0;
        bd[labels[i]].spinesum = 0;

}


__global__
void allMultiScan(spinerec *td, spinerec *bd, int *ld, int *vd, int root)
{
        int t = threadIdx.x + (blockIdx.x * blockDim.x);

	// spinetree phase
        for (int r = root - 1; r >= 0; r--)
        {
                int i = r * root + t;
                td[i].spine = bd[ld[i]].spine;
                __syncthreads();
                bd[ld[i]].spine = &td[i];
                __syncthreads();
        }


	int nextValue = vd[root * t];
	spinerec *nextSpine = td[root * t].spine;
	// rowsum phase
        for (int c = 1; c <= root; c++)
        {
                int i = c + (root * t);
		int value = nextValue;
		spinerec *curSpine = nextSpine;
		if (c != root)
		{
			nextValue = vd[i];
			nextSpine = &(*(td[i].spine));
                }
		curSpine->rowsum += value;
                __syncthreads();
        }

	// spinesum phase
	// we can leave out row 0
        for (int r = 1; r < root; r++)
        {
                int i = r * root + t;
		// should this if be in there or is it faster without?
                if (td[i].rowsum != 0)
                        td[i].spine->spinesum = td[i].spinesum + td[i].rowsum;
                __syncthreads();
        }

	nextValue = vd[root * t];
	nextSpine = td[root * t].spine;
	// multisum phase
        for (int c = 0; c < root; c++)
        {
                int i = c + (root * t);
		int value = nextValue;
		spinerec *curSpine = nextSpine;
		if (c != root - 1)
		{
			nextValue = vd[i+1];
			nextSpine = td[i+1].spine;
		}
		int spineSum = curSpine->spinesum;
                ld[i] = spineSum;
                __syncthreads();
                curSpine->spinesum = value + spineSum;
                __syncthreads();
        }

}

int main(int argc, char** argv)
{
        cudaFuncSetCacheConfig(initialize, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(allMultiScan, cudaFuncCachePreferL1);

	int root = 32;

        if (argc > 1)
                root = 32 * atoi(argv[1]);
        printf("Calculating multiscan for input size %i.\n", root * root);

        int m = 2;//64*64;
        int n = root * root;

        int *values;
        int *labels;

        int *result;

        int bucketSize = m * sizeof(spinerec);
        int tempSize = n * sizeof(spinerec);

        int inoutSize = n * sizeof(int);
	
        values = (int*) malloc(inoutSize);
        labels = (int*) malloc(inoutSize);
        result = (int*) malloc(inoutSize);

        if (result == 0 || labels == 0 || values == 0)
                printf("Error during host malloc.\n");

        int *vd;
        int *ld;
        spinerec *bd;
        spinerec *td;

        for (int i = 0; i < n; i++)
        {
                values[i] = i;
                labels[i] = rand() % m;
                result[i] = 0;
        }

        // distribute data into blocks
        // root is dividable by 32, therefore n by 1024
	int ndiv = 8;
	while (n / ndiv > 1024)
		ndiv *= 2;

        int rootdiv = 32;
        for (int i = 2; i < 33; i++)
        {
                if (root % (i * 32) == 0 && root / rootdiv > 32)
                        rootdiv = i * 32;
        }

        printf("Spine stage has %i blocks with %i threads each.\n", ndiv, n / ndiv);
        printf("Root stage has %i blocks with %i threads each.\n", root / rootdiv, rootdiv);

	cudaEvent_t startT;
        cudaEventCreate(&startT);

        cudaEvent_t stopT;
        cudaEventCreate(&stopT);

	cudaEventRecord(startT, NULL);	

        if (cudaSuccess != cudaMalloc( (void**) &vd, inoutSize))
		printf("Error during cudaMalloc vd.\n");
        if (cudaSuccess != cudaMalloc( (void**) &ld, inoutSize))
		printf("Error during cudaMalloc ld.\n");
        if (cudaSuccess != cudaMalloc( (void**) &bd, bucketSize))
		printf("Error during cudaMalloc bd.\n");
        if (cudaSuccess != cudaMalloc( (void**) &td, tempSize))
		printf("Error during cudaMalloc td.\n");
        if (cudaSuccess != cudaMemcpy(vd, values, inoutSize, cudaMemcpyHostToDevice))
		printf("Error during cudaMemcpy vd.\n");
        if (cudaSuccess != cudaMemcpy(ld, labels, inoutSize, cudaMemcpyHostToDevice))
		printf("Error during cudaMemcpy ld.\n");

        // Allocate CUDA events that we'll use for timing
        cudaEvent_t start;
        cudaEventCreate(&start);

        cudaEvent_t stop;
        cudaEventCreate(&stop);

        // Record the start event
        cudaEventRecord(start, NULL);

        // do stuff

        initialize<<<ndiv, n / ndiv>>>(td, bd, ld);

        allMultiScan<<<root / rootdiv, rootdiv>>>(td, bd, ld, vd, root);

        // Record the stop event
        cudaEventRecord(stop, NULL);

        // Wait for the stop event to complete
        //cudaEventSynchronize(stop);

        cudaMemcpy(result, ld, inoutSize, cudaMemcpyDeviceToHost);

        cudaFree(vd);
        cudaFree(ld);
        cudaFree(bd);
        cudaFree(td);

	cudaEventRecord(stopT, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stop);
        // Wait for the stop event to complete
        cudaEventSynchronize(stopT);



        float msecTotal = 0.0f;
	float msecTotalT = 0.0f;
        cudaEventElapsedTime(&msecTotal, start, stop);
	cudaEventElapsedTime(&msecTotalT, startT, stopT);

        for (int i = 0; i < 20; i++)
        {
                printf("%u \t %u \t %u \n", values[i], labels[i], result[i]);
        }

        printf("Took %f microseconds, which is %f microseconds per value.\n", msecTotal, msecTotal / (float) n);
	printf("Including transfers, took %f microseconds, which is %f microseconds per value.\n", msecTotalT, msecTotalT / (float) n);

        return EXIT_SUCCESS;
}
