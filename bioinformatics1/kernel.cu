#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_ptr.h"
#include "thrust/device_malloc.h"
#include "thrust/device_free.h"
#include "thrust/device_vector.h"
#include "thrust/sort.h"
#include "thrust/iterator/constant_iterator.h"
#include "thrust/extrema.h"
#include "utils.h"
#include "timer.h"

#include <stdio.h>
#include <algorithm>
#include <exception>

struct wordKeys {
	char k1, k2, k3, k4, k5, k6, k7, k8, k9;
};

__global__ void FindAllWords1(char **d_out, const char *d_in, int size) {

	int id = threadIdx.x + blockDim.x * blockIdx.x;

	//if (id <= size - 9) {
	//	char newWord[10];
	//	for (int i = 0; i < 9; i++)
	//	{
	//		newWord[i] = d_in[id + i];
	//	}
	//	newWord[9] = '\0';
	//	memcpy(d_out[id], newWord, 10);
	//}

	if (id <= size - 9) {
		for (int i = 0; i < 9; i++)
		{
			d_out[id][i] = d_in[id + i];
		}
		d_out[id][9] = '\0';
	}
}

__global__ void FindAllWords2(char *d_out, const char *d_in, int size) {

	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id <= size - 9) {
		for (int i = 0; i < 9; i++)
		{
			int index = id * 9 + i;
			d_out[index] = d_in[id + i];
		}
	}
}

////__global__ void FindAllWords3(wordKeys **d_out, const char *d_in, int size) {
////
////	int id = threadIdx.x + blockDim.x * blockIdx.x;
////
////	if (id <= size - 9) {
////		d_out[id]->k1 = d_in[id];
////		d_out[id]->k2 = d_in[id + 1];
////		d_out[id]->k3 = d_in[id + 2];
////		d_out[id]->k4 = d_in[id + 3];
////		d_out[id]->k5 = d_in[id + 4];
////		d_out[id]->k6 = d_in[id + 5];
////		d_out[id]->k7 = d_in[id + 6];
////		d_out[id]->k8 = d_in[id + 7];
////		d_out[id]->k9 = d_in[id + 8];
////	}
////}

struct sort_wordKeys {
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

struct equal_wordKeys {
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


struct charArrayCompare {
	__host__ __device__ bool operator()(const char* o1, const char* o2) {
		while ((*o1) && (*o2) && (*o1 == *o2))
		{
			o1++;
			o2++;
		}
		int diff = *(const unsigned char*)o1 - *(const unsigned char*)o2;
		return diff < 0;
	}
};

struct charArrayEqual {
	__host__ __device__ bool operator()(const char* o1, const char* o2) {
		while ((*o1) && (*o2))
		{
			if (*o1 != *o2)
				return false;
			o1++;
			o2++;
		}
		return true;
	}
};


void FMFW1(const char* const in, const int& sizeWord, const unsigned int size) {

	clock_t tStart1 = clock();
	clock_t tStart2;

	const int N_WORDS = size - sizeWord + 1;
	const int ARRAY_BYTES = size * sizeof(char);
	const int WORD_BYTES = sizeWord + 1 * sizeof(char);
	char* d_in = nullptr;
	char** d_listWords = nullptr;
	////char** h_listWords = nullptr;
	
	try {
		int maxThreads = size;
		int blocks = 1;

		if (size > 1024) {
			maxThreads = 1024;
			blocks = size / maxThreads;
		}

		checkCudaErrorsAndExit(cudaSetDevice(0));
		checkCudaErrors(cudaMalloc((void **)&d_in, ARRAY_BYTES));
		checkCudaErrors(cudaMemcpy(d_in, in, ARRAY_BYTES, cudaMemcpyHostToDevice));
		
		tStart2 = clock();

		//!1
		checkCudaErrors(cudaMallocManaged((void **)&d_listWords, N_WORDS * sizeof(char*)));
		for (int i=0; i < N_WORDS; i++)
		{
			checkCudaErrors(cudaMallocManaged((void**)&(d_listWords[i]), WORD_BYTES));
		}

		//!2
		////checkCudaErrors(cudaMalloc((void **)&d_listWords, N_WORDS * sizeof(char*)));
		////checkCudaErrors(cudaHostAlloc((void **)&h_listWords, N_WORDS * sizeof(char*), cudaHostAllocDefault));
		////for (int i = 0; i < N_WORDS; i++) 
		////{
		////	checkCudaErrors(cudaMalloc((void **)&h_listWords[i], WORD_BYTES));
		////}
		////checkCudaErrors(cudaMemcpy(d_listWords, h_listWords, N_WORDS * sizeof(char*), cudaMemcpyHostToDevice));
	
		FindAllWords1 <<<blocks, maxThreads>>> (d_listWords, d_in, size);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());		
	}
	catch (cudaError_t e)
	{
		cudaFree(d_in);
		cudaFree(d_listWords);
		////for (int i = 0; i < N_WORDS; i++) 
		////{
		////	cudaFreeHost(h_listWords[i]);
		////}		
		////cudaFreeHost(h_listWords);
		return;
	};

	//!Sort and reduce with Thrust.........

	thrust::device_vector<char*>dvector_listWords(N_WORDS);
	for (int i = 0; i < N_WORDS; i++) {
		dvector_listWords[i] = d_listWords[i];
	}

	std::cout << dvector_listWords.size() << std::endl;

	thrust::device_vector<char*>d_Words(N_WORDS);
	thrust::device_vector<int>d_Frequency(N_WORDS);

	thrust::sort(dvector_listWords.begin(), dvector_listWords.end(), charArrayCompare());

	thrust::pair<thrust::device_vector<char*>::iterator, thrust::device_vector<int>::iterator> new_end;
	new_end = thrust::reduce_by_key(dvector_listWords.begin(), dvector_listWords.end(),
				thrust::constant_iterator<int>(1), d_Words.begin(), d_Frequency.begin(), charArrayEqual());

	d_Words.erase(new_end.first, d_Words.end());
	d_Frequency.erase(new_end.second, d_Frequency.end());

	int rsize = new_end.first - d_Words.begin();

	thrust::sort_by_key(d_Frequency.begin(), d_Frequency.end(), d_Words.begin(), thrust::greater<int>());

	thrust::device_vector<int>::iterator d_MostFrequent = thrust::max_element(d_Frequency.begin(), d_Frequency.end());

	//Print most frequent words
	std::vector<char*> h_MostFrequentWords;
	for (int i = 0; i < rsize; i++) {
		if (d_Frequency[i] != *d_MostFrequent) {
			break;
		}
		char* w = d_Words[i];
		std::cout << w << std::endl;
		h_MostFrequentWords.push_back(w);
	}

	cudaFree(d_in);
	cudaFree(d_listWords);
	//for (int i = 0; i < N_WORDS; i++) {
	//	cudaFree(h_listWords[i]);
	//}
	//cudaFreeHost(h_listWords);

	printf("The parallel FMFW1 ran in %d ticks: %f secs.\n", clock() - tStart1, ((double)(clock() - tStart1)) / CLOCKS_PER_SEC);
	printf("The parallel FMFW1 ran in %d ticks: %f secs (transferDataHostToDev not included).\n\n", clock() - tStart2, ((double)(clock() - tStart2)) / CLOCKS_PER_SEC);
}

//!THE FASTEST SOLUTION
//!AND THE ONLY ONE THAT ALLOW TO ALLOCATE ENOUGH MEMORY
void FMFW2(const char* const in, const int& sizeWord, const unsigned int size) {

	clock_t tStart = clock();
	clock_t tStart1;

	const int N_WORDS = size - sizeWord + 1;
	const int ARRAY_BYTES = size * sizeof(char);
	char* d_in = nullptr;
	char* d_listWords = nullptr;
	char* h_out = nullptr; // (char*)malloc(N_WORDS * sizeWord * sizeof(char));

	try {

		int maxThreads = size;
		int blocks = 1;

		if (size > 1024) {
			maxThreads = 1024;			
			blocks = std::ceil((float)size / maxThreads);
		}

		checkCudaErrorsAndExit(cudaSetDevice(0));
		checkCudaErrors(cudaMalloc((void **)&d_in, ARRAY_BYTES));
		checkCudaErrors(cudaMemcpy(d_in, in, ARRAY_BYTES, cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((void **)&d_listWords, N_WORDS * sizeWord * sizeof(char)));
		checkCudaErrors(cudaHostAlloc((void **)&h_out, N_WORDS * sizeWord * sizeof(char), cudaHostAllocDefault));

		FindAllWords2<<<blocks, maxThreads>>>(d_listWords, d_in, size);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaMemcpy(h_out, d_listWords, N_WORDS * sizeWord * sizeof(char), cudaMemcpyDeviceToHost));
	}
	catch (cudaError_t e)
	{
		cudaFree(d_in);
		cudaFree(d_listWords);
		cudaFreeHost(h_out);
		return;
	};

	//!Sort and reduce with Thrust.........
	tStart1 = clock();

	thrust::device_vector<wordKeys>dvector_listWords(N_WORDS);
	for (int i = 0; i < N_WORDS; i++) {
		wordKeys w = {
			*(h_out + (i * sizeWord)), *(h_out + (i * sizeWord) + 1), *(h_out + (i * sizeWord) + 2),
			*(h_out + (i * sizeWord) + 3), *(h_out + (i * sizeWord) + 4), *(h_out + (i * sizeWord) + 5),
			*(h_out + (i * sizeWord) + 6), *(h_out + (i * sizeWord) + 7), *(h_out + (i * sizeWord) + 8)
		};
		//char w[10];
		//memcpy(w, h_out + (i * sizeWord), sizeWord + 1);
		//w[9] = '\0';
		dvector_listWords[i] = w;
		//std::cout << w << std::endl;
	}

	std::cout << dvector_listWords.size() << std::endl;

	int serialDelay = clock() - tStart1;

	thrust::device_vector<wordKeys>d_Words(N_WORDS);
	thrust::device_vector<int>d_Frequency(N_WORDS);

	thrust::sort(dvector_listWords.begin(), dvector_listWords.end(), sort_wordKeys());

	thrust::pair<thrust::device_vector<wordKeys>::iterator, thrust::device_vector<int>::iterator> new_end;
	new_end = thrust::reduce_by_key(dvector_listWords.begin(), dvector_listWords.end(),
		thrust::constant_iterator<int>(1), d_Words.begin(), d_Frequency.begin(), equal_wordKeys());

	d_Words.erase(new_end.first, d_Words.end());
	d_Frequency.erase(new_end.second, d_Frequency.end());

	int rsize = new_end.first - d_Words.begin();

	thrust::sort_by_key(d_Frequency.begin(), d_Frequency.end(), d_Words.begin(), thrust::greater<int>());

	thrust::device_vector<int>::iterator d_MostFrequent = thrust::max_element(d_Frequency.begin(), d_Frequency.end());

	//Print most frequent words
	std::vector<wordKeys> h_MostFrequentWords;
	for (int i = 0; i < rsize; i++) {
		if (d_Frequency[i] != *d_MostFrequent) {
			break;
		}
		wordKeys w = d_Words[i];
		std::cout << w.k1 << w.k2 << w.k3 << w.k4 << w.k5 << w.k6 << w.k7 << w.k8 << w.k9 << " " << d_Frequency[i] << std::endl;
		h_MostFrequentWords.push_back(w);
	}

	cudaFree(d_in);
	cudaFree(d_listWords);
	cudaFreeHost(h_out);


	printf("The parallel FMFW2 ran in %d ticks: %f secs.\n", clock() - tStart, ((double)(clock() - tStart)) / CLOCKS_PER_SEC);
	printf("Delay serial code is %d ticks: %f secs.\n\n", serialDelay, (double)serialDelay / CLOCKS_PER_SEC);
}


