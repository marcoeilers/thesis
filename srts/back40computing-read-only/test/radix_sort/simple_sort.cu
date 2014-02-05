/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 ******************************************************************************/

/******************************************************************************
 * Simple test driver program for radix sort.
 ******************************************************************************/

#include <stdio.h> 
#include <algorithm>

// Sorting includes
#include <b40c/radix_sort/enactor.cuh>
#include <b40c/util/multi_buffer.cuh>

// Test utils
#include "b40c_test_util.h"


struct Foo
{
	int a[5];
};

template<typename ValueType, int START_BIT, int NO_BITS>
void doSort(int*, ValueType*, int, bool);

/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	typedef int KeyType;
	typedef double ValueType;

    unsigned int num_elements = 1 << 26;


	// Allocate host problem data
    KeyType *h_keys = new KeyType[num_elements];
    ValueType *h_values = new ValueType[num_elements];
//	KeyType *h_reference_keys = new KeyType[num_elements];

	// Initialize host problem data

	for (int i = 0; i < num_elements; ++i)
	{
		b40c::util::RandomBits(h_keys[i]);
		h_values[i] = 1.0;
//		h_reference_keys[i] = h_keys[i];
	}

	doSort<ValueType, 0, sizeof(int)>(h_keys, h_values, num_elements, false);

	delete(h_keys);
	delete(h_values);
}

template<typename ValueType, int START_BIT, int NO_BITS>
void doSort(int *h_keys, ValueType *h_values, int num_elements, bool keys_only)
{
	typedef int KeyType;
	
	// Allocate device data. (We will let the sorting enactor create
	// the "pong" storage if/when necessary.)
	KeyType *d_keys;
	ValueType *d_values;
	cudaMalloc((void**) &d_keys, sizeof(KeyType) * num_elements);
	cudaMalloc((void**) &d_values, sizeof(ValueType) * num_elements);

	// Copy host data to device data
	cudaMemcpy(d_keys, h_keys, sizeof(KeyType) * num_elements, cudaMemcpyHostToDevice);
	cudaMemcpy(d_values, h_values, sizeof(ValueType) * num_elements, cudaMemcpyHostToDevice);

	// Create a reusable sorting enactor
	b40c::radix_sort::Enactor enactor;

	if (keys_only) {

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<KeyType> double_buffer;

		// The current key buffer (double_buffer.d_keys[double_buffer.selector]) backs the keys.
		double_buffer.d_keys[double_buffer.selector] = d_keys;

		// Allocate pong buffer
		cudaMalloc((void**) &double_buffer.d_keys[double_buffer.selector ^ 1], sizeof(KeyType) * num_elements);

		// Sort
		enactor.Sort(double_buffer, num_elements);
//		enactor.OneRunSort<START_BIT, NO_BITS>(double_buffer, num_elements);

		// Check keys answer
		printf("Simple keys-only sort:\n\n");

		// Cleanup "pong" storage
		if (double_buffer.d_keys[double_buffer.selector ^ 1]) {
			cudaFree(double_buffer.d_keys[double_buffer.selector ^ 1]);
		}

	} else {

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<KeyType, ValueType> double_buffer;

		// The current key buffer (double_buffer.d_keys[double_buffer.selector]) backs the keys.
		double_buffer.d_keys[double_buffer.selector] = d_keys;
		double_buffer.d_values[double_buffer.selector] = d_values;

		// Allocate pong buffer
		cudaMalloc((void**) &double_buffer.d_keys[double_buffer.selector ^ 1], sizeof(KeyType) * num_elements);
		cudaMalloc((void**) &double_buffer.d_values[double_buffer.selector ^ 1], sizeof(ValueType) * num_elements);

		// Sort
		enactor.Sort(double_buffer, num_elements);
//		enactor.OneRunSort<START_BIT, NO_BITS>(double_buffer, num_elements);
//printf("doing the oneRunSort call\n");
		// Check keys answer
		printf("Simple key-value sort:\n\n: ");

		// Copy out values
		cudaMemcpy(h_values, double_buffer.d_values[double_buffer.selector], sizeof(ValueType) * num_elements, cudaMemcpyDeviceToHost);



		// Cleanup "pong" storage
		if (double_buffer.d_keys[double_buffer.selector ^ 1]) {
			cudaFree(double_buffer.d_keys[double_buffer.selector ^ 1]);
		}
		if (double_buffer.d_values[double_buffer.selector ^ 1]) {
			cudaFree(double_buffer.d_values[double_buffer.selector ^ 1]);
		}

	}
	cudaFree(d_keys);
	cudaFree(d_values);
}

