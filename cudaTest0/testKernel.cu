#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
int atomicAdd(int* address, int val);
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "GenericCUDA.h"


__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void cubeKernel(int *d_out, const int *d_in)
{
	int i = threadIdx.x;
	int f = d_in[i];
	d_out[i] = f * f * f;
}

cudaError_t cubeWithCuda(int *out, const int *in, unsigned int size) {

	int *d_in = 0;
	int *d_out = 0;
	cudaError_t cudaStatus = cudaSuccess;

	try
	{
		std::cout << &d_in << std::endl;

		// Choose which GPU to run on, change this on a multi-GPU system.
		checkCudaErrorsAndExit(cudaSetDevice(0));

		transferDataHostToDev<int>(in, &d_in, &d_out, size);

		// Launch a kernel on the GPU 
		runKernel<int>(&d_out, &d_in, cubeKernel, size);

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(out, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			throw cudaStatus;
		}
	}
	catch (cudaError_t ex)
	{
		cudaStatus = ex;
		cudaFree(d_in);
		cudaFree(d_out);
	}
	cudaFree(d_in);
	cudaFree(d_out);

	return cudaStatus;
}