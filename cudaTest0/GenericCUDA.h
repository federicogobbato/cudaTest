#pragma once

#include "utils.h"
#include "timer.h"
#include <type_traits>

template <class T>
	//class = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value>::type>
void runKernel(T **d_out, T **d_in, void(*kernel)(T*, const T*),
	dim3 threads = dim3(1, 1, 1), dim3 blocks = dim3(1, 1, 1), size_t sharedMemorySize = 0)
{
	cudaError_t cudaStatus;

	if (sharedMemorySize > 0) {
		// Launch a kernel on the GPU with one thread for each element, that use shared memorys.
		(*kernel) <<<blocks, threads, sharedMemorySize>>> ((*d_out), (*d_in));
	}
	else {
		// Launch a kernel on the GPU with one thread for each element.
		(*kernel) <<<blocks, threads>>> ((*d_out), (*d_in));
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		throw cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		throw cudaStatus;
	}
}


template <class T>
	//class = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value>::type>
void transferDataHostToDev(const T *in, T **d_in, T **d_out, unsigned int sizeIN, unsigned int sizeOUT = 0)
{
	const int ARRAY_BYTES = sizeIN * sizeof(T);
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		throw cudaStatus;
	}

	// Allocate GPU buffers for 2 vectors (one input, one output)    .
	cudaStatus = cudaMalloc(d_in, ARRAY_BYTES);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc d_in failed!");
		throw cudaStatus;
	}

	cudaStatus = cudaMalloc(d_out, sizeOUT > 0 ? sizeOUT * sizeof(T) : ARRAY_BYTES);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc d_out failed!");
		throw cudaStatus;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(*d_in, in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy on d_in failed!");
		throw cudaStatus;
	}
}
