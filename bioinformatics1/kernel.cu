
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_ptr.h"
#include "thrust/device_malloc.h"
#include "thrust/device_free.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "thrust/iterator/constant_iterator.h"
#include "utils.h"

#include <stdio.h>
#include <algorithm>

struct wordKeys {
	char k1, k2, k3, k4, k5, k6, k7, k8, k9;
};

__global__ void FindAllWords(char *d_out, const char *d_in, int size) {

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	if (id <= size - 9) {
		for (int i = 0; i < 9; i++)
		{
			int index = (9 * id) + i;
			d_out[index] = d_in[id + i];
		}
	}
}

struct sort_functor {
	__host__ __device__ bool operator()(wordKeys &keys1, wordKeys &keys2) {
		if (keys1.k1 < keys2.k1) return true;
		if (keys1.k1 > keys2.k1) return false;
		if (keys1.k2 < keys2.k2) return true;
		if (keys1.k2 > keys2.k2) return false;
		if (keys1.k3 < keys2.k3) return true;
		if (keys1.k3 > keys2.k3) return false;
		if (keys1.k4 < keys2.k4) return true;
		if (keys1.k4 > keys2.k4) return false;
		if (keys1.k5 < keys2.k5) return true;
		if (keys1.k5 > keys2.k5) return false;
		if (keys1.k6 < keys2.k6) return true;
		if (keys1.k6 > keys2.k6) return false;
		if (keys1.k7 < keys2.k7) return true;
		if (keys1.k7 > keys2.k7) return false;
		if (keys1.k8 < keys2.k8) return true;
		if (keys1.k8 > keys2.k8) return false;
		if (keys1.k9 < keys2.k9) return true;
		return false;
	}
};

struct equal_key {
	__host__ __device__ bool operator()(wordKeys keys1, wordKeys keys2) {
		if ((keys1.k1 == keys2.k1) && 
			(keys1.k2 == keys2.k2) &&
			(keys1.k3 == keys2.k3) &&
			(keys1.k4 == keys2.k4) && 
			(keys1.k5 == keys2.k5) &&
			(keys1.k6 == keys2.k6) && 
			(keys1.k7 == keys2.k7) &&
			(keys1.k8 == keys2.k8) &&
			(keys1.k9 == keys2.k9)) return true;
		return false;
	}
};


bool CharArrayCompare(const char* o1, const char* o2) {
	while (*o1 && (*o1 == *o2))
	{
		o1++;
		o2++;
	}
	int diff = *(const unsigned char*)o1 - *(const unsigned char*)o2;
	return diff < 0;
	//return strcmp(o1, o2) < 0;
}


void FMFW(const char* const in, const int& sizeWord, const unsigned int size) {

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaSetDevice(0));

	const int ARRAY_BYTES = size * sizeof(char);
	char* d_in = nullptr;
	char* d_listWords = nullptr;
	char* h_out = (char*)malloc(ARRAY_BYTES * sizeWord);


	try {

		int maxThreads = size;
		int blocks = 1;

		if (size > 1024) {
			maxThreads = 1024;
			blocks = size / maxThreads;
		}

		checkCudaErrors(cudaMalloc((void **)&d_in, ARRAY_BYTES));
		checkCudaErrors(cudaMemcpy(d_in, in, ARRAY_BYTES, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void **)&d_listWords, ARRAY_BYTES * sizeWord));

		FindAllWords <<<blocks, maxThreads>>> (d_listWords, d_in, size);
		checkCudaErrorsAndExit(cudaDeviceSynchronize());
		checkCudaErrorsAndExit(cudaGetLastError());

		checkCudaErrorsAndExit(cudaMemcpy(h_out, d_listWords, ARRAY_BYTES * sizeWord, cudaMemcpyDeviceToHost));

		//Sort and reduce with Thrust.........
	}
	catch (cudaError_t ex)
	{
		cudaFree(d_in);
		cudaFree(d_listWords);
		free(h_out);
		return;
	};

	std::vector<wordKeys> hvector_listWords;
	for (int i = 0; i <= size - sizeWord; i++) {
		wordKeys k = {
			*(h_out + (i * sizeWord)), *(h_out + (i * sizeWord) + 1), *(h_out + (i * sizeWord) + 2),
			*(h_out + (i * sizeWord) + 3), *(h_out + (i * sizeWord) + 4), *(h_out + (i * sizeWord) + 5),
			*(h_out + (i * sizeWord) + 6), *(h_out + (i * sizeWord) + 7), *(h_out + (i * sizeWord) + 8)
		};

		hvector_listWords.push_back(k);
	}

	thrust::device_vector<wordKeys>dvector_listWords(hvector_listWords.begin(), hvector_listWords.end());
	thrust::device_vector<wordKeys>d_Words(size);
	thrust::device_vector<int>d_Frequency(size);

	thrust::sort(dvector_listWords.begin(), dvector_listWords.end(), sort_functor());

	thrust::pair<thrust::device_vector<wordKeys>::iterator, thrust::device_vector<int>::iterator> new_end;
	new_end = thrust::reduce_by_key(dvector_listWords.begin(), dvector_listWords.end(),
				thrust::constant_iterator<int>(1), d_Words.begin(), d_Frequency.begin(), equal_key());

	int rsize = new_end.first - d_Words.begin();

	std::cout << rsize << std::endl;

	//Print most frequent words
	for (int i = 0; i < rsize; i++) {
		wordKeys w = d_Words[i];
		int f = d_Frequency[i];
		if (f > 1)
		std::cout << w.k1 << w.k2 << w.k3 << w.k4 << w.k5 << w.k6 << w.k7 << w.k8 << w.k9 << " " << f << std::endl;
	}

	cudaFree(d_in);
	cudaFree(d_listWords);
	free(h_out);
}


