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


// Helper function for using CUDA to add vectors in parallel.
////cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
////{
////	int *dev_inA = 0;
////	int *dev_inB = 0;
////	int *dev_outC = 0;
////	cudaError_t cudaStatus;
////
////	// Choose which GPU to run on, change this on a multi-GPU system.
////	cudaStatus = cudaSetDevice(0);
////	if (cudaStatus != cudaSuccess) {
////		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
////		goto Error;
////	}
////
////	// Allocate GPU buffers for three vectors (two input, one output)    .
////	cudaStatus = cudaMalloc((void**)&dev_outC, size * sizeof(int));
////	if (cudaStatus != cudaSuccess) {
////		fprintf(stderr, "cudaMalloc failed!");
////		goto Error;
////	}
////
////	cudaStatus = cudaMalloc((void**)&dev_inA, size * sizeof(int));
////	if (cudaStatus != cudaSuccess) {
////		fprintf(stderr, "cudaMalloc failed!");
////		goto Error;
////	}
////
////	cudaStatus = cudaMalloc((void**)&dev_inB, size * sizeof(int));
////	if (cudaStatus != cudaSuccess) {
////		fprintf(stderr, "cudaMalloc failed!");
////		goto Error;
////	}
////
////	// Copy input vectors from host memory to GPU buffers.
////	cudaStatus = cudaMemcpy(dev_inA, a, size * sizeof(int), cudaMemcpyHostToDevice);
////	if (cudaStatus != cudaSuccess) {
////		fprintf(stderr, "cudaMemcpy failed!");
////		goto Error;
////	}
////
////	cudaStatus = cudaMemcpy(dev_inB, b, size * sizeof(int), cudaMemcpyHostToDevice);
////	if (cudaStatus != cudaSuccess) {
////		fprintf(stderr, "cudaMemcpy failed!");
////		goto Error;
////	}
////
////	// Launch a kernel on the GPU with one thread for each element.
////	addKernel <<<1, size >>> (dev_outC, dev_inA, dev_inB);
////
////	// Check for any errors launching the kernel
////	cudaStatus = cudaGetLastError();
////	if (cudaStatus != cudaSuccess) {
////		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
////		goto Error;
////	}
////
////	// cudaDeviceSynchronize waits for the kernel to finish, and returns
////	// any errors encountered during the launch.
////	cudaStatus = cudaDeviceSynchronize();
////	if (cudaStatus != cudaSuccess) {
////		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
////		goto Error;
////	}
////
////	// Copy output vector from GPU buffer to host memory.
////	cudaStatus = cudaMemcpy(c, dev_outC, size * sizeof(int), cudaMemcpyDeviceToHost);
////	if (cudaStatus != cudaSuccess) {
////		fprintf(stderr, "cudaMemcpy failed!");
////		goto Error;
////	}
////
////Error:
////	cudaFree(dev_outC);
////	cudaFree(dev_inA);
////	cudaFree(dev_inB);
////
////	return cudaStatus;
////}