#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <bitset>
#include "Cache.h"
#include "AssocCache.h"
#include "RAM.h"
#include <sys/time.h>
#include <inttypes.h>
#include <iomanip>


using namespace std;

#define SIMULATE

#define NO_RUNS 4
#define MIN_EXP 20
#define MAX_EXP 24

#ifdef SIMULATE
#define COPY(A, B) mem->copy(&A, &B)
#define GET(A) mem->fetchAddress(&A)
#define STORE(A, B) mem->storeAddress(&A, B)
#define INCR(A) STORE(A, GET(A) + 1)
#define DECR(A) STORE(A, GET(A) - 1)
#else
#define COPY(A, B) B = A
#define GET(A) A
#define STORE(A, B) A = B
#define INCR(A) ++A
#define DECR(A) --A
#endif

#define NO_BITS 7

#define ONE_BITS ((1 << NO_BITS) - 1)
#define ByteOf(x) (((x) >> bitsOffset) & ONE_BITS)
#define BteOf(x, y) (((x) >> y) & ONE_BITS)

uint64_t totalCost;

static void getCounts(int runs, int startRun, int counts[][1 << NO_BITS], Memory* mem, int size, int *values)
{
  int *cp = &counts[0][0];
  for (int i = 2 * runs * (1 << NO_BITS); i > 0; --i, ++cp)
  {
      //*cp = 0;
      STORE((*cp), 0);
  }


  int *sp = values;
  for (int i = size; i > 0; --i, ++sp)
  {
    for (int j = 0; j < runs; j++)
    {
      cp = (&counts[2*j][0]) + BteOf(*sp, (j+startRun) * NO_BITS);
      //++(*cp);
      INCR((*cp));
      cp = (&counts[2*j+1][0]) + BteOf(*sp, (j+startRun) * NO_BITS);
      INCR((*cp));
    }
  }

}


static void radix (int counts[][1 << NO_BITS], Memory* mem, int run, short bitsOffset, int size, int *values, int *result)
{
	int *cp, *sp;

	// scan
	int sum;
	sum = 0;
	cp = counts[2 * run];
	for (int i = (1 << NO_BITS); i > 0; --i, ++cp) {
		int cur = GET(*cp);
		//*cp = sum;
		STORE((*cp), sum);
		// also scan mirror array
		int *cpNext = cp + (1 << NO_BITS);
		STORE((*cpNext), sum);
		sum += cur;
	}

	sp = values;
	for (int i = size; i > 0; --i, ++sp) {
		int cur = GET((*sp));

		int* ct = counts[2 * run];
        cp = ct + ByteOf (cur);
		int ind = *cp;
		//result[ind] = cur;
		STORE(result[ind], cur);
        INCR((*cp));
	}
}

void simpleGather(Memory* mem, int *values, int *indices, int *result, int size)
{
    for (int i = 0; i < size; i++)
    {
        int index = GET(indices[i]);
        int value = GET(values[index]);
        STORE(result[i], value);
    }
}

void printAll(std::string title, int* data, int size, bool bin = false)
{
    cout << title << "\n";
    for (int i = 0; i < size; i++)
    {
        if (bin)
        {
            bitset<8> x(data[i]);
            cout << data[i] << "(" << x << ") ";
        }
        else
            cout << data[i] << " ";
    }
    cout << "\n";
}

void getValuesFromSorted(Memory* mem, int* dest, int counts[][1 << NO_BITS], int run, short bitsOffset, int* valSource, int *indexSource, int size)
{
    int *countsPtr = counts[2 * run + 1];
    for (int i = size; i > 0; --i, ++dest, ++indexSource)
    {
        int cur = GET((*indexSource));
        int *indexPtr = countsPtr + ByteOf(cur);
        int value = GET(valSource[*indexPtr]);
        //*dest = source[*indexPtr];
        STORE((*dest), value);
        INCR((*indexPtr));
    }
}

void sortGather(Memory* mem, int *values, int *indices, int *result, int *temp1, int *temp2, int *temp3, int size, int exponent)
{
    int runs = exponent / NO_BITS;
    cout << runs << " runs \n";
    int counts[2 * runs][1 << NO_BITS];

    getCounts(runs, 1, counts, mem, size, indices);
//    printAll("counts 0", counts[0], 1 << NO_BITS);

    // sort
    for (int i = 0; i < runs; i++)
        radix(counts, mem, i, (i+1) * NO_BITS, size, i == 0 ? indices : i == 1 ? temp1 : temp2, i == 0 ? temp1 : i == 1 ? temp2 : temp3);


    // get values
    for (int i = 0; i < size; i++)
    {
        int *curArray = runs == 1 ? temp1 : runs == 2 ? temp2 : temp3;
        int index = GET(curArray[i]);
        int value = GET(values[index]);
        STORE(curArray[i], value);
    }

    // put em back
    for (int i = runs - 1; i >= 0; i--)
        getValuesFromSorted(mem, i == 0 ? result : i == 1 ? temp1 : temp2, counts, i, (i+1) * NO_BITS, i == 0 ? temp1 : i == 1 ? temp2 : temp3, i == 0 ? indices : i == 1 ? temp1 : temp2, size);


}


int main()
{
    RAM* ram = new RAM(&totalCost, 240);
    AssocCache<32768, 16>* l2 = new AssocCache<32768, 16>(ram, &totalCost, 35, 2048 * 1024 * 8 / (64 * 8), 4096 * 1024 * 8);
    AssocCache<512, 8>* l1 = new AssocCache<512, 8>(l2, &totalCost, 2, 32 * 1024 * 8 / (64 * 8), 32 * 1024 * 8);


//for (int power = 8; power < 17; power++){
    //int exponent = 26;
    for (int exponent = MIN_EXP; exponent <= MAX_EXP; exponent++)
{
    totalCost = 0;
    int size = 1 << exponent;
    int* values = (int*) calloc(size, sizeof(int));
    int* indices = (int*) calloc(size, sizeof(int));
    int* temp1 = (int*) calloc(size, sizeof(int));
    int* temp2 = (int*) calloc(size, sizeof(int));
    int* temp3 = (int*) calloc(size, sizeof(int));
    int* result1 = (int*) calloc(size, sizeof(int));
    int* result2 = (int*) calloc(size, sizeof(int));

/*
    // test cache
    Memory* mem = l1;
    for (int i = 0; i < size; i++)
    {
        values[i] = rand() % 3000;
        indices[i] = i;
        temp1[i] = rand() % size;
    }

    simpleGather(mem, values, indices, result1, size);
    float seqMiss = l1->missPercentage();
    l1->resetStatistics();
    int seqCost = totalCost;
    simpleGather(mem, values, temp1, result2, size);
    float randomMiss = l1->missPercentage();
    int randomCost = totalCost - seqCost;
    cout << "sequential\t" << seqCost << "\t" << seqMiss << "\trandom\t" << randomCost << "\t" << randomMiss << "\n";
    */

    for (int i = 0; i < size; i++)
    {
        values[i] = rand() % 100;
        indices[i] = rand() % size;
        temp1[i] = 15;
        temp2[i] = 15;
        temp3[i] = 15;
        result1[i] = 15;
        result2[i] = 15;
    }

//    printAll("indices", indices, size, true);
    timeval before, between, after;
    gettimeofday(&before, NULL);
    int runs = 4;
    for (int i = 0; i < NO_RUNS; i++)
{
/*        int counts[2 * runs][1 << NO_BITS];

        getCounts(runs, 1, counts, l1, size, indices);

        // sort
        for (int i = 0; i < runs; i++)
            radix(counts, l1, i, (i+1) * NO_BITS, size, i == 0 ? indices : i == 1 ? temp1 : temp2, i == 0 ? temp1 : i == 1 ? temp2 : temp3);
*/

    simpleGather(l1, values, indices, result1, size);
}
    float simpleMiss1 = l1->missPercentage();
    float simpleMiss2 = l2->missPercentage();
    l1->resetStatistics();
    l2->resetStatistics();
    uint64_t time = totalCost;
    gettimeofday(&between, NULL);
    for (int i = 0; i < NO_RUNS; i++)
        sortGather(l1, values, indices, result2, temp1, temp2, temp3, size, exponent);
    gettimeofday(&after, NULL);
    uint64_t time2 = totalCost - time;
    float sortMiss1 = l1->missPercentage();
    float sortMiss2 = l2->missPercentage();


//    for (int i = 0; i < size; i++)
//        assert(result1[i] == result2[i]);

#ifdef SIMULATE
    cout << size << "\tsimpleGather\t" << time / NO_RUNS << "\t" << simpleMiss1 << "\t" << simpleMiss2 << "\tsortGather\t" << time2 / NO_RUNS << "\t" << sortMiss1 << "\t" << sortMiss2 << "\n";
#else
    float simpleTime = ((between.tv_sec - before.tv_sec) * 1e9 + (between.tv_usec - before.tv_usec) * 1e3) / NO_RUNS;
    float sortTime = (1e9 * (after.tv_sec - between.tv_sec) + 1e3 * (after.tv_usec - between.tv_usec)) / NO_RUNS;
    cout << before.tv_sec << " " << between.tv_sec << " " << after.tv_sec << "\n";
    cout << before.tv_usec << " " << between.tv_usec << " " << after.tv_usec << "\n";

    cout << size << "\tsimpleGather\t" << fixed << setprecision(3) << simpleTime<<  "\tsortGather\t" << sortTime << "\n";
#endif

    free(indices);
    free(values);
    free(temp1);
    free(temp2);
    free(temp3);
    free(result1);
    free(result2);
}



    return 0;
}
