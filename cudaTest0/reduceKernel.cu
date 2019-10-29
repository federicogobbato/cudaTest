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
#include "helper_timer.h"


//!Seems to be the faster solution for reduce
template<class T>
__global__ void reduceKernel1(T *d_out, const T *d_in)
{
	extern __shared__ T sdata[];

	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	// load shared mem from global mem
	sdata[tid] = d_in[myId];
	__syncthreads();            // make sure entire block is loaded!

	int tmp = sdata[tid];
	__syncthreads();
	atomicAdd(&sdata[0], tmp);
	__syncthreads();

	// only thread 0 writes result for this block back to global mem
	if (tid == 0)
	{
		d_out[blockIdx.x] = sdata[0] - d_in[(myId / blockDim.x)*blockDim.x];
	}
}

template<class T>
__global__ void reduceKernel2(T *d_out, const T *d_in)
{
	extern __shared__ T sdata[];

	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	sdata[tid] = d_in[myId];
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

cudaError_t reduceWithCuda(unsigned int *out, unsigned const int *in, unsigned int size) {

	unsigned int* d_in = 0;
	unsigned int* d_intermediate = 0;
	unsigned int* d_out = 0;
	cudaError_t cudaStatus = cudaSuccess;

	try
	{
		int maxThreads = size;
		int blocks = 1;

		if (size > 1024) {
			maxThreads = 1024;
			blocks = size / maxThreads;
		}

		transferDataHostToDev<unsigned int>(in, &d_in, &d_intermediate, size, blocks);

		GpuTimer* timer = new GpuTimer();
		////StopWatchWin *timerPro = new StopWatchWin();
		timer->Start();
		////timerPro->start();

		// Launch a kernel on the GPU 
		runKernel<unsigned int>(&d_intermediate, &d_in, reduceKernel1, maxThreads, blocks, maxThreads * sizeof(int));

		while (blocks > 1024) {
			blocks /= 1024;
			checkCudaErrors(cudaMalloc((void **)&d_out, blocks * sizeof(int)));

			runKernel<unsigned int>(&d_out, &d_intermediate, reduceKernel1, maxThreads, blocks, maxThreads * sizeof(int));				
			checkCudaErrors(cudaMemcpy(d_intermediate, d_out, blocks * sizeof(int), cudaMemcpyDeviceToDevice));
			cudaFree(d_out);
		}

		checkCudaErrors(cudaMalloc((void **)&d_out, sizeof(int)));
		runKernel<unsigned int>(&d_out, &d_intermediate, reduceKernel1, blocks, 1, blocks * sizeof(int));

		timer->Stop();
		////timerPro->stop();
		printf("(CUDA Timer) The parallel code ran in: %f msecs (transferDataHostToDev not included).\n", timer->Elapsed());
		////float reduceTime = timerPro->getAverageTime() * 1e-3;
		////printf("(CUDA PRO timer) Throughput = %.4f GB/s, Time = %.5f sec \n", 
		////	1.0e-9 * ((double)(size * sizeof(unsigned int))) / reduceTime, reduceTime);
		////delete timerPro;

		// Copy output vector from GPU buffer to host memory.
		checkCudaErrors(cudaMemcpy(out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
	}
	catch (cudaError_t ex)
	{
		cudaStatus = ex;
		cudaFree(d_in);
		cudaFree(d_out);
		cudaFree(d_intermediate);
	};

	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_intermediate);

	return cudaStatus;
}