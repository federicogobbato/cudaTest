#include <curand.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>

#include "utils.h"
#include "timer.h"
#include "helper_timer.h"


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t cubeWithCuda(int *c, const int *a, unsigned int size);
cudaError_t reduceWithCuda(unsigned int *c, unsigned const int *a, unsigned int size);

unsigned int* GenerateRandomNumber(size_t n);

int main()
{
    srand(time(NULL));
	const int arraySize = 1 << 26;
	const int byteSize = arraySize * sizeof(unsigned int);
	printf("Size array %d \n", arraySize);
	unsigned int* h_in = new unsigned int[arraySize];
	unsigned int h_out;


	//--------------------------------------------------------------------------------
	printf("START SERIAL RANDOM GENERATION \n");
	clock_t tStart = clock();	
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
	thrust::device_vector<unsigned int>* temp = new  thrust::device_vector<unsigned int>(h_in, h_in + arraySize);
	tStart = clock();
	int result1 = thrust::reduce(temp->begin(), temp->end(), (unsigned int)0, thrust::plus<unsigned int>());
	printf("The parallel code ran in (thrust) %d ticks: %f secs.\n", 
		clock() - tStart, ((double)(clock() - tStart)) / CLOCKS_PER_SEC);
	printf("result %d \n\n", result1);
	delete temp;

	//--------------------------------------------------------------------------------
	printf("START PARALLEL RECUCE \n");
	cudaDeviceSynchronize();
	tStart = clock();	
	cudaError_t cudaStatus = reduceWithCuda(&h_out, h_in, arraySize);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAIL!");
		exit(1);
	}

	printf("The parallel code ran in %d ticks: %f secs.\n", 
		clock() - tStart, ((double)(clock() - tStart)) / CLOCKS_PER_SEC);
	printf("result %d\n\n", h_out);

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

