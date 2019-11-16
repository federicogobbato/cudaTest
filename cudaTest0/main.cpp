#include <curand.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>

#include "utils.h"
#include "timer.h"

const int NUM_SEGMENTS = 8;

cudaError_t cubeWithCuda(int *c, const int *a, unsigned int size);
cudaError_t reduceWithCuda(unsigned int *out, unsigned int *in, int size);
cudaError_t reduceWithCuda(unsigned int *out, unsigned int **in, int nSegment, int sizeSegment, int size);
cudaError_t reduceWithCudaEnd(unsigned int *out, unsigned int **in, int nSegment, int sizeSegment, int size);

unsigned int* GenerateRandomNumber(size_t n);

int main()
{
	cudaFree(0);
    srand(time(NULL));
	clock_t tStart;
	const long arraySize = 1 << 26;
	const long byteSize = arraySize * sizeof(unsigned int);
	printf("Size array %d \n", arraySize);

	//--------------------------------------------------------------------------------
	printf("HOST MALLOC \n");
	tStart = clock();
	unsigned int* h_in = new unsigned int[arraySize];
	unsigned int h_out1, h_out2, h_out3;
	printf("Malloc for host memory ran in %d ticks: %f secs.\n\n",
		clock() - tStart, ((double)(clock() - tStart)) / CLOCKS_PER_SEC);


	//--------------------------------------------------------------------------------
	printf("START SERIAL RANDOM GENERATION \n");
	tStart = clock();
	for (int i = 0; i < arraySize; i++) {
		int r = std::rand();
		h_in[i] = r % 10 + 1;
	}
	printf("The random serial generation ran in %d ticks: %f secs.\n\n", 
		clock() - tStart, ((double)(clock() - tStart)) / CLOCKS_PER_SEC);

	//--------------------------------------------------------------------------------
	printf("START PARALLEL RANDOM GENERATION \n");
	cudaDeviceSynchronize();
	tStart = clock();
	unsigned int* h_inCuda = GenerateRandomNumber(arraySize);
	printf("The random parallel generation ran in %d ticks: %f secs.\n\n", 
		clock() - tStart, ((double)(clock() - tStart)) / CLOCKS_PER_SEC);


	//--------------------------------------------------------------------------------
	printf("START SERIAL REDUCE \n");
	tStart = clock();
	int result0 = h_in[0];
	for (int i = 1; i < arraySize; i++) {
		result0 += h_in[i];
	}
	printf("The serial code ran in %d ticks: %f secs.\n", 
		clock() - tStart, ((double)(clock() - tStart)) / CLOCKS_PER_SEC);
	printf("expected result %d \n\n", result0);	

	//--------------------------------------------------------------------------------
	printf("START PARALLEL REDUCE (thrust)\n");

	tStart = clock();
	thrust::device_vector<unsigned int>* temp = new thrust::device_vector<unsigned int>(h_in, h_in + arraySize);	
	//thrust::device_vector<unsigned int> temp(h_in, h_in + arraySize);

	int result1 = thrust::reduce(temp->begin(), temp->end(), (unsigned int)0, thrust::plus<unsigned int>());
	printf("The parallel code ran in (thrust) %d ticks: %f secs.\n", 
		clock() - tStart, ((double)(clock() - tStart)) / CLOCKS_PER_SEC);
	printf("result %d \n\n", result1);
	delete temp;


	////--------------------------------------------------------------------------------
	printf("START PARALLEL RECUCE 1 \n");
	checkCudaErrorsAndExit(cudaDeviceSynchronize());

	tStart = clock();
	cudaError_t cudaStatus = reduceWithCuda(&h_out1, h_in, arraySize);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAIL!");
		goto FREE;
	}

	printf("The parallel code ran in %d ticks: %f secs.\n",
		clock() - tStart, ((double)(clock() - tStart)) / CLOCKS_PER_SEC);
	printf("result %d\n\n", h_out1);


	//--------------------------------------------------------------------------------
	printf("START PARALLEL RECUCE 2 \n");
	checkCudaErrorsAndExit(cudaDeviceSynchronize());

	unsigned int** h_inPinned = new unsigned int*[NUM_SEGMENTS];
	// Allocate host pagable memory on host pinned memory (cudaMallocHost has a limit!!!)
	const long sizeOneSegment = std::ceil((float)arraySize / NUM_SEGMENTS);
	for (int i = 0; i < NUM_SEGMENTS; i++) {
		checkCudaErrorsAndExit(cudaMallocHost((void**)&h_inPinned[i], sizeOneSegment * sizeof(unsigned int)));
		int sizeToCopy = sizeOneSegment;
		if (i == NUM_SEGMENTS - 1)
		{
			sizeToCopy = arraySize - sizeOneSegment * i;
		}
		memcpy((void*)h_inPinned[i], h_in + sizeOneSegment * i, sizeToCopy * sizeof(unsigned int));
	}	

	tStart = clock();	
	cudaStatus = reduceWithCuda(&h_out2, h_inPinned, NUM_SEGMENTS, sizeOneSegment, arraySize);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAIL!");
		goto FREE;
	}

	printf("The parallel code ran in %d ticks: %f secs.\n", 
		clock() - tStart, ((double)(clock() - tStart)) / CLOCKS_PER_SEC);
	printf("result %d\n\n", h_out2);


	//--------------------------------------------------------------------------------
	printf("START PARALLEL RECUCE 3 \n");
	checkCudaErrorsAndExit(cudaDeviceSynchronize());

	tStart = clock();
	cudaStatus = reduceWithCudaEnd(&h_out3, h_inPinned, NUM_SEGMENTS, sizeOneSegment, arraySize);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAIL!");
		goto FREE;
	}

	printf("The parallel code ran in %d ticks: %f secs.\n",
		clock() - tStart, ((double)(clock() - tStart)) / CLOCKS_PER_SEC);
	printf("result %d\n\n", h_out3);


	FREE:
	cudaFreeHost(h_inPinned);
	delete h_in;
	free(h_inCuda);
	
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

    return 0;
}


unsigned int* GenerateRandomNumber(size_t n) {
	size_t i;
	curandGenerator_t gen;
	unsigned int *devData, *hostData;

	/* Allocate n floats on host */
	hostData = (unsigned int *)calloc(n, sizeof(unsigned int));

	/* Allocate n floats on device */
	checkCudaErrorsAndExit(cudaMalloc((void **)&devData, n * sizeof(unsigned int)));

	/* Create pseudo-random number generator */
	curandStatus_t mess = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	if (mess != CURAND_STATUS_SUCCESS) printf("Error at %s:%d\n", __FILE__, __LINE__);

	/* Set seed */
	mess = curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	if (mess != CURAND_STATUS_SUCCESS) printf("Error at %s:%d\n", __FILE__, __LINE__);

	/* Generate n floats on device */
	curandGenerate(gen, devData, n);

	/* Copy device memory to host */
	checkCudaErrorsAndExit(cudaMemcpy(hostData, devData, n * sizeof(unsigned int),
		cudaMemcpyDeviceToHost));

	/* Cleanup */
	mess = curandDestroyGenerator(gen);
	if (mess != CURAND_STATUS_SUCCESS) printf("Error at %s:%d\n", __FILE__, __LINE__); 

	checkCudaErrorsAndExit(cudaFree(devData));

	return hostData;
}

