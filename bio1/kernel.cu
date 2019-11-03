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

__global__ void FindAllWords1(char **d_out, const char *d_in, int size, int sizeWord) {

	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id < size)
	{
		char* newWord = (char*)malloc(sizeWord * sizeof(char)); 
		for (int i = 0; i < sizeWord; i++)
		{
			//d_out[id][i] = d_in[id + i];
			newWord[i] = d_in[id + i];
		}
		//d_out[id][sizeWord] = '\0';
		newWord[sizeWord] = '\0';
		d_out[id] = newWord;
	}
}

__global__ void CopyWords(char **d_in, char** d_out) {

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	printf("%s \n", d_in[id]);
	memcpy(d_out[id], d_in[id], 10);
}

__global__ void FreeWords(char **d_in, int size) {

	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id < size)
	{
		free(d_in[id]);
	}
}

__global__ void FindAllWordsShared1(char **d_out, const char *d_in, int size, int sizeWord) {

	const int RADIUS = sizeWord;

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int tindex = threadIdx.x;

	extern __shared__ char temp[];
	temp[tindex] = d_in[id];
	if (threadIdx.x < RADIUS) {
		temp[tindex + blockDim.x] = d_in[id + blockDim.x];
	}
	__syncthreads();

	if (id < size)
	{
		//?d_out[id] = (char*)malloc(sizeWord * sizeof(char));
		for (int i = 0; i < sizeWord; i++)
		{			
			d_out[id][i] = temp[tindex + i];
		}	
		d_out[id][sizeWord] = '\0';
	}
}


__global__ void FindAllWordsShared2(char *d_out, const char *d_in, int size, int sizeWord) {

	const int RADIUS = sizeWord;

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int tindex = threadIdx.x;

	extern __shared__ char temp[];
	temp[tindex] = d_in[id];
	if (threadIdx.x < RADIUS) {
		temp[tindex + blockDim.x] = d_in[id + blockDim.x];
	}
	__syncthreads();

	if (id < size)
	{
		for (int i = 0; i < sizeWord; i++)
		{
			d_out[(id * sizeWord) + i] = temp[tindex + i];
		}
	}
}


////__global__ void FindAllWords2(wordKeys **d_out, const char *d_in, int size) {
////
////	int id = threadIdx.x + blockDim.x * blockIdx.x;
////
////	if (id < size) {
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
////
////struct sort_wordKeys {
////	__host__ __device__ bool operator()(wordKeys &keys1, wordKeys &keys2) {
////		if (keys1.k1 < keys2.k1) return true;
////		if (keys1.k1 > keys2.k1) return false;
////		if (keys1.k2 < keys2.k2) return true;
////		if (keys1.k2 > keys2.k2) return false;
////		if (keys1.k3 < keys2.k3) return true;
////		if (keys1.k3 > keys2.k3) return false;
////		if (keys1.k4 < keys2.k4) return true;
////		if (keys1.k4 > keys2.k4) return false;
////		if (keys1.k5 < keys2.k5) return true;
////		if (keys1.k5 > keys2.k5) return false;
////		if (keys1.k6 < keys2.k6) return true;
////		if (keys1.k6 > keys2.k6) return false;
////		if (keys1.k7 < keys2.k7) return true;
////		if (keys1.k7 > keys2.k7) return false;
////		if (keys1.k8 < keys2.k8) return true;
////		if (keys1.k8 > keys2.k8) return false;
////		if (keys1.k9 < keys2.k9) return true;
////		return false;
////	}
////};
////
////struct equal_wordKeys {
////	__host__ __device__ bool operator()(wordKeys keys1, wordKeys keys2) {
////		if ((keys1.k1 == keys2.k1) && 
////			(keys1.k2 == keys2.k2) &&
////			(keys1.k3 == keys2.k3) &&
////			(keys1.k4 == keys2.k4) && 
////			(keys1.k5 == keys2.k5) &&
////			(keys1.k6 == keys2.k6) && 
////			(keys1.k7 == keys2.k7) &&
////			(keys1.k8 == keys2.k8) &&
////			(keys1.k9 == keys2.k9)) return true;
////		return false;
////	}
////};


struct charArrayCompare {
	__host__ __device__ bool operator()(const char* o1, const char* o2) {
		while ((*o1) && (*o1 == *o2))
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
		while (*o1)
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

	clock_t tStart = clock();
	clock_t tDelay;
	int Delay = 0;

	const int N_WORDS = size - sizeWord + 1;
	const int ARRAY_BYTES = size * sizeof(char);
	const int WORD_BYTES = (sizeWord + 1) * sizeof(char);
	char* d_in = nullptr;
	char** d_listWords = nullptr;

	char** h_MostFrequentWords = nullptr;
	int N_MFWORDS = 0;

	checkCudaErrorsAndExit(cudaSetDevice(0));

	try {

		int maxThreads = size;
		int blocks = 1;

		if (size > 1024) {
			maxThreads = 1024;
			blocks = std::ceil((float)size / maxThreads);
		}

		checkCudaErrors(cudaMalloc((void **)&d_in, ARRAY_BYTES));
		checkCudaErrors(cudaMemcpy(d_in, in, ARRAY_BYTES, cudaMemcpyHostToDevice));

		tDelay = clock();

		checkCudaErrors(cudaMalloc((void **)&d_listWords, N_WORDS * sizeof(char*)));
		////checkCudaErrors(cudaHostAlloc((void **)&d_listWords, N_WORDS * sizeof(char*), cudaMemoryTypeManaged));
		////for (int i = 0; i < N_WORDS; i++)
		////{
		////	checkCudaErrors(cudaHostAlloc((void**)&(d_listWords[i]), WORD_BYTES, cudaMemoryTypeManaged));
		////}

		Delay = clock() - tDelay;

		//FindAllWordsShared1 <<<blocks, maxThreads, (maxThreads + sizeWord) * sizeof(char)>>> (d_listWords, d_in, N_WORDS, sizeWord);
		FindAllWords1 << <blocks, maxThreads >>> (d_listWords, d_in, N_WORDS, sizeWord);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		//!Sort and reduce with Thrust.........

		//!Use a device_vector SLOW a lot the process, better use device_ptr
		thrust::device_ptr<char*> dev_ptr(d_listWords);
		thrust::sort(dev_ptr, dev_ptr + N_WORDS, charArrayCompare());

		thrust::device_vector<char*>d_Words(N_WORDS);
		thrust::device_vector<int>d_Frequency(N_WORDS);

		thrust::pair<thrust::device_vector<char*>::iterator, thrust::device_vector<int>::iterator> new_end;
		new_end = thrust::reduce_by_key(dev_ptr, dev_ptr + N_WORDS, thrust::constant_iterator<int>(1), d_Words.begin(), d_Frequency.begin(), charArrayEqual());

		int rsize = new_end.first - d_Words.begin();

		d_Words.erase(new_end.first, d_Words.end());
		d_Frequency.erase(new_end.second, d_Frequency.end());

		thrust::sort_by_key(d_Frequency.begin(), d_Frequency.end(), d_Words.begin(), thrust::greater<int>());

		thrust::device_vector<int>::iterator d_MostFrequent = thrust::max_element(d_Frequency.begin(), d_Frequency.end());

		//Print most frequent words
		thrust::device_ptr<char*> d_MostFrequentWords = thrust::device_malloc(rsize);
		N_MFWORDS = 0;
		for (int i = 0; i < rsize; i++) {
			if (d_Frequency[i] != *d_MostFrequent) {
				break;
			}
			d_MostFrequentWords[i] = d_Words[i];
			N_MFWORDS++;
		}

		checkCudaErrors(cudaHostAlloc((void **)&h_MostFrequentWords, N_MFWORDS * sizeof(char*), cudaMemoryTypeManaged));
		for (int i = 0; i < N_WORDS; i++)
		{
			checkCudaErrors(cudaHostAlloc((void**)&(h_MostFrequentWords[i]), WORD_BYTES, cudaMemoryTypeManaged));
		}

		CopyWords <<<1, N_MFWORDS >>> (thrust::raw_pointer_cast(d_MostFrequentWords), h_MostFrequentWords);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		////checkCudaErrors(cudaMemcpy(h_MostFrequentWords, thrust::raw_pointer_cast(d_MostFrequentWords), N_MFWORDS * sizeof(char*), cudaMemcpyDeviceToHost));
		////for (int i = 0; i < N_WORDS; i++)
		////{
		////	checkCudaErrors(cudaMemcpy(h_MostFrequentWords[i], thrust::raw_pointer_cast(d_MostFrequentWords)[i], N_MFWORDS * sizeof(char*), cudaMemcpyDeviceToHost));
		////}

		for (int i = 0; i < N_MFWORDS; i++) {
			printf("%s \n", h_MostFrequentWords[i]);
		}

		FreeWords <<<blocks, maxThreads >>> (d_listWords, N_WORDS);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	catch (cudaError_t e)
	{
		cudaFree(d_in);
		cudaFree(d_listWords);
		for (int i = 0; i < N_MFWORDS; i++)
			cudaFreeHost(h_MostFrequentWords[i]);
		cudaFreeHost(h_MostFrequentWords);
		return;
	};

	cudaFree(d_in);
	cudaFree(d_listWords);

	printf("The parallel FMFW2 ran in %d ticks: %f secs.\n", clock() - tStart - Delay, ((double)(clock() - tStart - Delay)) / CLOCKS_PER_SEC);
	printf("Delay code is %d ticks: %f secs.\n\n", Delay, (double)Delay / CLOCKS_PER_SEC);
}

