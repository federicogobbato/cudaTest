#pragma once

#ifdef __INTELLISENSE__
void __syncthreads();
int atomicAdd(int* address, int val);
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "GenericCUDA.h"
#include "utils.h"
#include "timer.h"

const int MAX_THREADS_BLOCK = 256;

template<class T>
__global__ void reduceKernel(T *d_out, const T *d_in, const int size)
{
	extern __shared__ T sdata[];

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	if (id < size) {
		// load shared mem from global mem and, make sure entire block is loaded!
		sdata[tid] = d_in[id];
		__syncthreads();

		atomicAdd(&sdata[0], sdata[tid]);
		__syncthreads();

		// only thread 0 writes result for this block back to global mem
		if (tid == 0)
		{
			int blockResult = sdata[0] - d_in[(id / blockDim.x)*blockDim.x];
			d_out[blockIdx.x] = blockResult;
		}
	}
}

//!Seems to be the faster solution for reduce but WORK ONLY for int?!
template<class T>
__global__ void reduceKernel1(T *d_out, const T *d_in, const int size)
{
	extern __shared__ T sdata[];

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	if (id < size) {
		// load shared mem from global mem and, make sure entire block is loaded!
		sdata[tid] = d_in[id];
		__syncthreads();            

		atomicAdd(&sdata[0], sdata[tid]);
		__syncthreads();

		if (tid == 0)
		{
			int blockResult = sdata[0] - d_in[(id / blockDim.x)*blockDim.x];
			atomicAdd(d_out, blockResult);
		}
	}
}

template<class T>
__global__ void reduceKernel2(T *d_out, const T *d_in, int size)
{
	extern __shared__ T sdata[];

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	if (id < size) {
		sdata[tid] = d_in[id];
		__syncthreads();

		for (unsigned int s = 1; s < blockDim.x; s *= 2) {
			if (tid % (2 * s) == 0) {
				sdata[tid] += sdata[tid + s];
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			d_out[blockIdx.x] = sdata[0];
		}
	}
}

template<class T>
__global__ void reduceKernel3(T *d_out, const T *d_in)
{
	extern __shared__ T sdata[];

	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	sdata[tid] = d_in[myId];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		d_out[blockIdx.x] = sdata[0];
	}
}

cudaError_t reduceWithCuda(unsigned int *out, unsigned int *in, int size) {

	const int ARRAY_BYTES = size * sizeof(unsigned int);
	unsigned int* d_in = 0;
	unsigned int* d_out = 0;
	////unsigned int* d_intermediate = 0;
	cudaError_t cudaStatus = cudaSuccess;

	GpuTimer* timer = new GpuTimer();
	double delay = 0;
	double kernelTime = 0;

	try
	{
		// Choose which GPU to run on, change this on a multi-GPU system.
		checkCudaErrorsAndExit(cudaSetDevice(0));

		int maxThreads = size;
		int blocks = 1;

		if (size > MAX_THREADS_BLOCK) {
			maxThreads = MAX_THREADS_BLOCK;
			blocks = std::ceil((float)size / maxThreads);
		}

		// Allocate GPU buffers for 2 vectors (one input, one output)    .
		checkCudaErrors(cudaMalloc((void**)&d_in, ARRAY_BYTES));
		////checkCudaErrors(cudaMalloc((void**)&d_intermediate, ARRAY_BYTES));
		checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(unsigned int)));

		timer->Start();
		// Copy input vectors from host memory to GPU buffers.
		checkCudaErrors(cudaMemcpy(d_in, in, ARRAY_BYTES, cudaMemcpyHostToDevice));
		timer->Stop();
		delay = timer->Elapsed();

		timer->Start();
		// Launch a kernel on the GPU 
		reduceKernel1 <<<blocks, maxThreads, maxThreads * sizeof(unsigned int) >>> (d_out, d_in, size);

		////reduceKernel <<<blocks, maxThreads, maxThreads * sizeof(unsigned int) >>> (d_intermediate, d_in, size);

		////while (blocks > maxThreads) {
		////	int maxSize = blocks;
		////	blocks = std::ceil((float)blocks / maxThreads);
		////	
		////	////checkCudaErrors(cudaMalloc((void **)&d_out, blocks * sizeof(unsigned int)));
		////	////reduceKernel <<<blocks, maxThreads, maxThreads * sizeof(unsigned int) >>> (d_out, d_intermediate, maxSize);
		////	////checkCudaErrors(cudaMemcpy(d_intermediate, d_out, blocks * sizeof(int), cudaMemcpyDeviceToDevice));
		////	////cudaFree(d_out);

		////	reduceKernel <<<blocks, maxThreads, maxThreads * sizeof(unsigned int) >>> (d_intermediate, d_intermediate, maxSize);
		////}

		////checkCudaErrors(cudaMalloc((void **)&d_out, sizeof(unsigned int)));
		////reduceKernel <<<1, blocks, blocks * sizeof(unsigned int) >>> (d_out, d_intermediate, blocks);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		timer->Stop();
		kernelTime = timer->Elapsed();

		// Copy output vector from GPU buffer to host memory.
		checkCudaErrors(cudaMemcpy(out, d_out, sizeof(unsigned int), cudaMemcpyDeviceToHost));

		printf("(CUDA Timer) Parallel code delay: %f secs.\n", delay / 1000);
		printf("(CUDA Timer) Parallel code kernel: %f secs.\n", kernelTime / 1000);
	}
	catch (cudaError_t ex)
	{
		cudaStatus = ex;
		goto FREE;
	};

FREE:
	cudaFree(d_in);
	cudaFree(d_out);
	////cudaFree(d_intermediate);
	delete timer;

	return cudaStatus;
}

cudaError_t reduceWithCuda(unsigned int *out, unsigned int **in, int nSegment, int sizeSegment, int size) {

	const int ARRAY_BYTES = sizeSegment * sizeof(unsigned int);
	unsigned int* d_in = 0;
	unsigned int* d_out = 0;
	unsigned int h_out = 0;
	cudaError_t cudaStatus = cudaSuccess;

	GpuTimer* timer = new GpuTimer();
	double delay = 0;
	double kernelTime = 0;

	try
	{
		// Choose which GPU to run on, change this on a multi-GPU system.
		checkCudaErrorsAndExit(cudaSetDevice(0));

		int maxThreads = sizeSegment;
		int blocks = 1;

		if (sizeSegment > MAX_THREADS_BLOCK) {
			maxThreads = MAX_THREADS_BLOCK;
			blocks = std::ceil((float)sizeSegment / maxThreads);
		}

	
		// Allocate GPU buffers for 2 vectors (one input, one output)    .
		checkCudaErrors(cudaMalloc((void**)&d_in, ARRAY_BYTES));
		checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(unsigned int)));

		for (int i = 0; i < nSegment; i++) 
		{
			timer->Start();
			// Copy input vectors from host memory to GPU buffers.
			checkCudaErrors(cudaMemcpy(d_in, in[i], ARRAY_BYTES, cudaMemcpyHostToDevice));
			timer->Stop();
			delay += timer->Elapsed();

			int maxSize = sizeSegment;
			if (i == nSegment - 1)
			{	
				maxSize = size - sizeSegment * i;
			}
		
			timer->Start();
			// Launch a kernel on the GPU 
			reduceKernel1 <<<blocks, maxThreads, maxThreads * sizeof(unsigned int) >>> (d_out, d_in, maxSize);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			timer->Stop();
			kernelTime += timer->Elapsed();

			// Copy output vector from GPU buffer to host memory.
			checkCudaErrors(cudaMemcpy(&h_out, d_out, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			std::cout << *out << std::endl;
			(*out) += h_out;
			checkCudaErrors(cudaMemset(d_out, 0, sizeof(unsigned int)));
		}

		printf("(CUDA Timer) Parallel code delay: %f secs.\n", delay / 1000);
		printf("(CUDA Timer) Parallel code kernel: %f secs.\n", kernelTime / 1000);
	}
	catch (cudaError_t ex)
	{
		cudaStatus = ex;
		goto FREE;
	};

FREE:
	cudaFree(d_in);
	cudaFree(d_out);
	delete timer;	

	return cudaStatus;
}


cudaError_t reduceWithCudaEnd(unsigned int *out, unsigned int **in, int nSegment, int sizeSegment, int size) {

	const int ARRAY_BYTES = sizeSegment * sizeof(unsigned int);
	unsigned int* d_in = 0;
	unsigned int* d_out = 0;
	unsigned int* d_intermediate = 0;
	unsigned int h_out = 0;
	cudaError_t cudaStatus = cudaSuccess;

	GpuTimer* timer = new GpuTimer();
	double delay = 0;
	double kernelTime = 0;

	try
	{
		// Choose which GPU to run on, change this on a multi-GPU system.
		checkCudaErrorsAndExit(cudaSetDevice(0));

		int THREADS = sizeSegment;
		int BLOCKS = 1;

		if (sizeSegment > MAX_THREADS_BLOCK) {
			THREADS = MAX_THREADS_BLOCK;
			BLOCKS = std::ceil((float)sizeSegment / THREADS);
		}


		// Allocate GPU buffers for 2 vectors (one input, one output)    .
		checkCudaErrors(cudaMalloc((void**)&d_in, ARRAY_BYTES));	
		checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(unsigned int)));
		checkCudaErrors(cudaMalloc((void**)&d_intermediate, ARRAY_BYTES));

		for (int i = 0; i < nSegment; i++)
		{
			int blocks = BLOCKS;

			timer->Start();
			// Copy input vectors from host memory to GPU buffers.
			checkCudaErrors(cudaMemcpy(d_in, in[i], ARRAY_BYTES, cudaMemcpyHostToDevice));
			timer->Stop();
			delay += timer->Elapsed();

			int maxSize = sizeSegment;
			if (i == nSegment - 1)
			{
				maxSize = size - sizeSegment * i;
			}

			timer->Start();
			// Launch a kernel on the GPU 
			reduceKernel2 <<<blocks, THREADS, THREADS * sizeof(unsigned int) >>> (d_intermediate, d_in, maxSize);

			while (blocks > THREADS) {
				int maxSize = blocks;
				blocks = std::ceil((float)blocks / THREADS);
				reduceKernel2 <<<blocks, THREADS, THREADS * sizeof(unsigned int) >>> (d_intermediate, d_intermediate, maxSize);
			}

			checkCudaErrors(cudaMalloc((void **)&d_out, sizeof(unsigned int)));
			reduceKernel2 <<<1, blocks, blocks * sizeof(unsigned int) >>> (d_out, d_intermediate, blocks);

			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			timer->Stop();
			kernelTime += timer->Elapsed();

			// Copy output vector from GPU buffer to host memory.
			checkCudaErrors(cudaMemcpy(&h_out, d_out, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			(*out) += h_out;
			checkCudaErrors(cudaMemset(d_out, 0, sizeof(unsigned int)));
			checkCudaErrors(cudaMemset(d_intermediate, 0, ARRAY_BYTES));
		}

		printf("(CUDA Timer) Parallel code delay: %f secs.\n", delay / 1000);
		printf("(CUDA Timer) Parallel code kernel: %f secs.\n", kernelTime / 1000);
	}
	catch (cudaError_t ex)
	{
		cudaStatus = ex;
		goto FREE;
	};

FREE:
	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_intermediate);
	delete timer;

	return cudaStatus;
}