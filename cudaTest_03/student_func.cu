/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#pragma once

#ifdef __INTELLISENSE__
void __syncthreads();
int atomicAdd(int* address, int val);
int atomicAdd(unsigned int* address, int val);
int atomicMax(int* address, int val);
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"
#include <algorithm> 

__global__ void findMin(const float* const d_in, float* const d_out, const size_t numCols)
{
	extern __shared__ float sdata[];

	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	sdata[tid] = d_in[index];
	__syncthreads();            

	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) 
		{			
			if (sdata[tid] > sdata[tid + s])
			{
				sdata[tid] = sdata[tid + s];
			}
		}
		__syncthreads();
	}

	if (tid == 0)
	{	
		d_out[blockIdx.x] = sdata[0];
	}
}


__global__ void findMax(const float* const d_in, float* const d_out, const size_t numCols)
{
	extern __shared__ float sdata[];

	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	sdata[tid] = d_in[index];
	__syncthreads();            

	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0)
		{
			if (sdata[tid] < sdata[tid + s])
			{
				sdata[tid] = sdata[tid + s];
			}
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		d_out[blockIdx.x] = sdata[0];
	}
}


__global__ void histo(const float* const d_in, unsigned int* const d_bins,
					  const float min, const float range, const int numBins) 
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	int binIndex = ((d_in[index] - min) / range) * numBins;
	__syncthreads();
	atomicAdd(&d_bins[binIndex], 1);
}

//! d_out will have the same size of d_in
//! later the array will be sorted and reduced; 
__global__ void histoNoAtomic(const float* const d_in, unsigned int* const d_out,
							  const float min, const float range, const int numBins)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	int binIndex = ((d_in[index] - min) / range) * numBins;
	d_out[index] = binIndex;
}

//! INCLUSIVE SCAN
__global__ void cdfInclusive(unsigned int* const d_bins, int shift)
{
	int tid = threadIdx.x;

	int tmp = d_bins[tid];
	__syncthreads();
	d_bins[tid + shift] += tmp;
}

__global__ void cdfShift(unsigned int* const d_bins)
{
	int tid = threadIdx.x;

	int tmp = d_bins[tid];
	__syncthreads();
	d_bins[tid + 1] = tmp;
}



void your_histogram_and_prefixsum(const float* const d_logLuminance,
								  unsigned int* const d_cdf,
								  float &min_logLum,
								  float &max_logLum,
								  const size_t numRows,
								  const size_t numCols,
								  const size_t numBins)
{
	int size = numCols * numRows;
	int threads = 1024;
	int blocks = size / threads;

	//! REDUCE
	float* d_minValues = 0;
	float* d_maxValues = 0;
	checkCudaErrors(cudaMalloc(&d_minValues, numCols * numRows * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_maxValues, numCols * numRows * sizeof(float)));
	findMin<<<blocks, threads, threads * sizeof(float)>>>(d_logLuminance, d_minValues, numCols);
	findMax<<<blocks, threads, threads * sizeof(float)>>>(d_logLuminance, d_maxValues, numCols);

	float* d_minValue;
	float* d_maxValue;
	checkCudaErrors(cudaMalloc(&d_minValue, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_maxValue, sizeof(float)));
	findMin<<<1, blocks, blocks * sizeof(float)>>> (d_minValues, d_minValue, numCols);
	findMax<<<1, blocks, blocks * sizeof(float)>>> (d_maxValues, d_maxValue, numCols);

	checkCudaErrors(cudaMemcpy(&min_logLum, d_minValue, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_logLum, d_maxValue, sizeof(float), cudaMemcpyDeviceToHost));
	float range = max_logLum - min_logLum;

	printf("\n %f %f\n", min_logLum, max_logLum);
	
	//? Later try don't use AtomicAdd
	histo<<<blocks, threads>>>(d_logLuminance, d_cdf, min_logLum, range, numBins);

	int shift = 1;
	while (shift < numBins)
	{
		cdfInclusive <<<1, numBins>>> (d_cdf, shift);
		shift *= 2;
	}

	cdfShift<<<1, numBins >>>(d_cdf);

  //TODO
  /*Here are the steps you need to implement
	1) find the minimum and maximum value in the input logLuminance channel
	   store in min_logLum and max_logLum
	2) subtract them to find the range
	3) generate a histogram of all the values in the logLuminance channel using
	   the formula: bin = ((lum[i] - lumMin) / lumRange) * numBins
	4) Perform an exclusive scan (prefix sum) on the histogram to get
	   the cumulative distribution of luminance values (this should go in the
	   incoming d_cdf pointer which already has been allocated for you)       */

	cudaFree(d_minValues);
	cudaFree(d_maxValues);
	cudaFree(d_minValue);
	cudaFree(d_maxValue);
	//cudaFree(bins);
}
