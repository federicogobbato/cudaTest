#include <curand.h>

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
	const int byteSize = arraySize * sizeof(int);
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
	printf("The random serial generation ran in: %f secs.\n\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);


	//--------------------------------------------------------------------------------
	printf("START PARALLEL RANDOM GENERATION \n");
	cudaDeviceSynchronize();
	StopWatchInterface *timerPro = 0;
	sdkCreateTimer(&timerPro);

	sdkStartTimer(&timerPro);
	tStart = clock();

	unsigned int* h_inCuda = GenerateRandomNumber(arraySize);
	sdkStopTimer(&timerPro);

	printf("The random parallel generation ran in: %f secs.\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	double reduceTime = sdkGetAverageTimerValue(&timerPro) * 1e-3;
	printf("(CUDA PRO timer) Throughput = %.4f GB/s, Time = %.5f sec \n\n", 1.0e-9 * ((double)byteSize) / reduceTime, reduceTime);


	//--------------------------------------------------------------------------------
	printf("START SERIAL REDUCE \n");
	tStart = clock();
	int result = h_inCuda[0];
	for (int i = 1; i < arraySize; i++) {
		result += h_inCuda[i];
	}
	printf("The serial code ran in: %f secs.\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	printf("expected result %d \n\n", result);	

	
	//--------------------------------------------------------------------------------
	printf("START PARALLEL RECUCE \n");
	cudaDeviceSynchronize();

	GpuTimer timer;
	timer.Start();
	sdkStartTimer(&timerPro);
	tStart = clock();	

	cudaError_t cudaStatus = reduceWithCuda(&h_out, h_inCuda, arraySize);
	timer.Stop();
	sdkStopTimer(&timerPro);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAIL!");
		exit(1);
	}
	printf("The custom code ran in: %f secs.\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	printf("The custom code ran in (CUDA Timer): %f msecs.\n", timer.Elapsed());
	reduceTime = sdkGetAverageTimerValue(&timerPro) * 1e-3;
	printf("(CUDA PRO timer) Throughput = %.4f GB/s, Time = %.5f sec \n", 1.0e-9 * ((double)byteSize) / reduceTime, reduceTime);
	printf("result %d\n\n", h_out);

	delete h_in;
	free(h_inCuda);
	sdkDeleteTimer(&timerPro);
	
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
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

	/* Set seed */
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

	/* Generate n floats on device */
	curandGenerate(gen, devData, n);

	/* Copy device memory to host */
	checkCudaErrorsAndExit(cudaMemcpy(hostData, devData, n * sizeof(unsigned int),
		cudaMemcpyDeviceToHost));

	/* Cleanup */
	curandDestroyGenerator(gen);
	checkCudaErrorsAndExit(cudaFree(devData));

	return hostData;
}

