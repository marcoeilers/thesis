#ifndef VALINDEX
#define VALINDEX

#include <thrust/functional.h>

struct valIndex
{
int value;
int index;
};


struct valIndexAdd : public thrust::binary_function<valIndex,valIndex,valIndex> {
  __host__ __device__ valIndex operator() (valIndex a, valIndex b) {
  valIndex result;
  result.value = a.value + b.value;
  result.index = b.index;
  return result;
  }
};

#endif

