
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
#include <exception>

struct wordKeys {
	char k1, k2, k3, k4, k5, k6, k7, k8, k9;
};

__global__ void FindAllWords1(char *d_out, const char *d_in, int size) {

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

__global__ void FindAllWords2(wordKeys **d_out, const char *d_in, int size) {

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	if (id <= size - 9) {
		d_out[id]->k1 = d_in[id];
		d_out[id]->k2 = d_in[id + 1];
		d_out[id]->k3 = d_in[id + 2];
		d_out[id]->k4 = d_in[id + 3];
		d_out[id]->k5 = d_in[id + 4];
		d_out[id]->k6 = d_in[id + 5];
		d_out[id]->k7 = d_in[id + 6];
		d_out[id]->k8 = d_in[id + 7];
		d_out[id]->k9 = d_in[id + 8];
	}
}

__global__ void FindAllWords3(char **d_out, const char *d_in, int size) {

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	if (id <= size - 9) {
		for (int i = 0; i < 9; i++)
		{
			d_out[id][i] = d_in[id + i];
		}
		d_out[id][9] = '\0';
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

	const int N_WORDS = size - sizeWord + 1;
	const int ARRAY_BYTES = size * sizeof(char);
	char* d_in = nullptr;

	//!FIRST solution
	////char* d_listWords = nullptr;
	////char* h_out = nullptr;

	//!SECOND solution 
	////wordKeys** d_listWords = nullptr;
		
	//!THIRD solution 
	char** d_listWords = nullptr;

	try {

		int maxThreads = size;
		int blocks = 1;

		if (size > 1024) {
			maxThreads = 1024;
			blocks = size / maxThreads;
		}

		checkCudaErrors(cudaMalloc((void **)&d_in, ARRAY_BYTES));
		checkCudaErrors(cudaMemcpy(d_in, in, ARRAY_BYTES, cudaMemcpyHostToDevice));

		//!FIRST solution
		////checkCudaErrors(cudaMalloc((void **)&d_listWords, N_WORDS * sizeWord * sizeof(char)));
		////checkCudaErrors(cudaHostAlloc((void **)&h_out, N_WORDS * sizeWord * sizeof(char), cudaHostAllocDefault));

		//!SECOND solution
		////checkCudaErrors(cudaMallocManaged((void **)&d_listWords, N_WORDS * sizeof(wordKeys*)));
		////for (int i = 0; i < N_WORDS; i++)
		////{
		////	checkCudaErrors(cudaMallocManaged((void**)&(d_listWords[i]), sizeof(wordKeys)));
		////}

		//!THIRD solution
		checkCudaErrors(cudaMallocManaged((void **)&d_listWords, N_WORDS * sizeof(char*)));
		for (int i = 0; i < N_WORDS; i++)
		{
			checkCudaErrors(cudaMallocManaged((void**)&(d_listWords[i]), sizeof(char) * sizeWord + 1));
		}

		FindAllWords3 <<<blocks, maxThreads>>> (d_listWords, d_in, size);

		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());

		char* word0 = d_listWords[0];
		std::cout << word0 << std::endl;

		//!FIRST solution
		////checkCudaErrors(cudaMemcpy(h_out, d_listWords, N_WORDS * sizeWord * sizeof(char), cudaMemcpyDeviceToHost));

	}
	catch (cudaError_t e)
	{
		cudaFree(d_in);
		cudaFree(d_listWords);
		////cudaFreeHost(h_out);
		return;
	};


	//!Sort and reduce with Thrust.........

	//!FIRST solution
	////std::vector<wordKeys> hvector_listWords;
	////for (int i = 0; i < N_WORDS; i++) {
	////	wordKeys w = {
	////		*(h_out + (i * sizeWord)), *(h_out + (i * sizeWord) + 1), *(h_out + (i * sizeWord) + 2),
	////		*(h_out + (i * sizeWord) + 3), *(h_out + (i * sizeWord) + 4), *(h_out + (i * sizeWord) + 5),
	////		*(h_out + (i * sizeWord) + 6), *(h_out + (i * sizeWord) + 7), *(h_out + (i * sizeWord) + 8)
	////	};
	////	std::cout << w.k1 << w.k2 << w.k3 << w.k4 << w.k5 << w.k6 << w.k7 << w.k8 << w.k9 << std::endl;
	////	hvector_listWords.push_back(w);
	////}
	////thrust::device_vector<wordKeys>dvector_listWords(hvector_listWords.begin(), hvector_listWords.end());

	//!SECOND solution
	////thrust::device_vector<wordKeys>dvector_listWords(N_WORDS);
	////for (int i = 0; i < N_WORDS; i++) {
	////	dvector_listWords[i] = *d_listWords[i];
	////	wordKeys w = dvector_listWords[i];
	////	std::cout << w.k1 << w.k2 << w.k3 << w.k4 << w.k5 << w.k6 << w.k7 << w.k8 << w.k9 << std::endl;
	////}

	//!THIRD solution
	thrust::device_vector<char*>dvector_listWords(N_WORDS);
	for (int i = 0; i < N_WORDS; i++) {
		dvector_listWords[i] = d_listWords[i];
		char* w = dvector_listWords[i];
		std::cout << w << std::endl;
	}

	//thrust::device_vector<wordKeys>d_Words(N_WORDS);
	//thrust::device_vector<int>d_Frequency(N_WORDS);

	//thrust::sort(dvector_listWords.begin(), dvector_listWords.end(), sort_functor());

	//thrust::pair<thrust::device_vector<wordKeys>::iterator, thrust::device_vector<int>::iterator> new_end;
	//new_end = thrust::reduce_by_key(dvector_listWords.begin(), dvector_listWords.end(),
	//			thrust::constant_iterator<int>(1), d_Words.begin(), d_Frequency.begin(), equal_key());

	//int rsize = new_end.first - d_Words.begin();

	//std::cout << rsize << std::endl;

	////Print most frequent words
	//for (int i = 0; i < rsize; i++) {
	//	wordKeys w = d_Words[i];
	//	int f = d_Frequency[i];
	//	if (f > 1)
	//	std::cout << w.k1 << w.k2 << w.k3 << w.k4 << w.k5 << w.k6 << w.k7 << w.k8 << w.k9 << " " << f << std::endl;
	//}

	cudaFree(d_in);
	cudaFree(d_listWords);
	//free(h_out);
	//cudaFreeHost(h_out);
}


