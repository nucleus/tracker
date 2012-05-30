/*
 * 	File: ForegroundSegmenter.cu
 * 	----------------------------
 * 	This file contains the CUDA implementations of foreground segmentation.
 * 
 *	Written by Michael Andersch, 2012. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cutil_inline.h>
#include <cmath>

#include "ForegroundSegmenter.h"

texture<uchar4, 2, cudaReadModeElementType> input;		// input image, unmodified
texture<float, 2, cudaReadModeElementType> inputProcessed;	// input image, grayscale and blurred
texture<float, 2, cudaReadModeElementType> background_tex;	// current background image

__device__ __constant__ float kernel[GAUSSIAN_WINDOW_RADIUS];

/*	Function: SegmentAndUpdateBackgroundKernel
 * 	------------------------------------------
 * 	Segments the preprocessed input image using the background model.
 * 	Output is a foreground binary image and an updated background model. 
 */
__global__ void SegmentAndUpdateBackgroundKernel(float* segmented, float* background, unsigned width, unsigned height, float rate) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned offset = y * width + x;
	
	if(x >= width || y >= height)
		return;
	
	float in_sample = tex2D(inputProcessed, x+0.5f, y+0.5f);
	float last_bg_sample = tex2D(background_tex, x+0.5f, y+0.5f);
	float new_bg_sample = last_bg_sample * (1.0-rate) + in_sample * rate;
	float binary;
	
	in_sample = fabs(in_sample - last_bg_sample);
	in_sample = in_sample* in_sample;

	if(in_sample > THRESH)
		binary = 255.0f;
	else
		binary = 0.0f;
	
	segmented[offset] = binary;
	background[offset] = new_bg_sample;
}

/*	Function: GaussianBlurHorizontalKernel
 * 	------------------------------------
 * 	The horizontal pass of the gaussian filtering operation.
 */
__global__ void GaussianBlurHorizontalKernel(float* dst, unsigned ksize, unsigned width, unsigned height) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned offset = y * width + x;
	
	if(x >= width || y >= height)
		return;
	
	float sum = 0;
	for(int i = 0; i < ksize; i++)
		sum += kernel[i] * tex2D(inputProcessed, x-ksize/2+i+0.5f, y+0.5f);
	dst[offset] = sum;
}	

/*	Function: GaussianBlurVerticalKernel
 * 	------------------------------------
 * 	The vertical pass of the gaussian filtering operation.
 */
__global__ void GaussianBlurVerticalKernel(float* dst, unsigned ksize, unsigned width, unsigned height) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned offset = y * width + x;
	
	if(x >= width || y >= height)
		return;
	
	float sum = 0;
	for(int i = 0; i < ksize; i++)
		sum += kernel[i] * tex2D(inputProcessed, x+0.5f, y-ksize/2+i+0.5f);
	dst[offset] = sum;
}

/*	Function: Rgb2GrayKernel
 * 	------------------------
 * 	This kernel converts the input image in texture memory to grayscale and stores it to dst.
 */
__global__ void Rgb2GrayKernel(float *dst, unsigned width, unsigned height) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned offset = y * width + x;
	
	if(x >= width || y >= height)
		return;

	uchar4 sample;
	sample = tex2D(input, x+0.5f, y+0.5f);
	float gray = sample.x * 0.2989f + 0.587f * sample.y + 0.114f * sample.z;
	dst[offset] = gray;
}

/*	Function: createKernel1D
 * 	------------------------
 * 	This function generates 1-dimensional gaussian kernels.
 */
void createKernel1D(unsigned ksize, string type) {
	if(ksize%2 != 1) {
		fprintf(stderr, "Bad kernel size in kernel creation, aborting");
		exit(EXIT_FAILURE);
	}
	
	float kern[GAUSSIAN_WINDOW_RADIUS];
	if(type == "gaussian") {
		double sigma_x = ksize/3.;
		for(int i=0;i<=ksize/2;i++){
			kern[i+ksize/2] = exp(-i*i/(2*sigma_x*sigma_x))/(sqrt(2*M_PI)*sigma_x);
		}
		for(int i=0;i<ksize/2;i++){
			kern[i] = kern[ksize-1-i];
		}
	} else {
		fprintf(stderr, "Bad kernel type in kernel creation, aborting");
		exit(EXIT_FAILURE);
	}
	cudaMemcpyToSymbol("kernel", kern, sizeof(kern));
}

/*	Function: preProcessImage
 * 	-------------------------
 * 	This function operates on an RGBA color image in the device. It will first launch a kernel to perform
 * 	RGBA to grayscale conversion (which also converts uchar4 to float). Then, a gaussian blur filter is
 * 	implemented using separation into horizontal and vertical passes. 
 */
void preProcessImage(uchar4* src, float* tmpGray, float* tmpGauss, float* final, unsigned width, unsigned height) {
	{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        if(cudaBindTexture2D(0, input, src, channelDesc, width, height, width*sizeof(uchar4)) != cudaSuccess) {
		fprintf(stderr, "Cannot bind input image texture!");
		return;
	}
        input.addressMode[0] = cudaAddressModeClamp;
        input.addressMode[1] = cudaAddressModeClamp;
	input.normalized = false;
	
	dim3 dimBlock(FS_BLOCKSIZE_X,FS_BLOCKSIZE_Y);
	dim3 dimGrid((width+dimBlock.x-1)/dimBlock.x, (height+dimBlock.y-1)/dimBlock.y);	
	Rgb2GrayKernel<<<dimGrid, dimBlock>>>(tmpGray, width, height);
	}
	
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	{
        if(cudaBindTexture2D(0, inputProcessed, tmpGray, channelDesc, width, height, width*sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Cannot bind post-grayscale conversion texture!");
		return;
	}
        inputProcessed.addressMode[0] = cudaAddressModeClamp;
        inputProcessed.addressMode[1] = cudaAddressModeClamp;
        inputProcessed.filterMode = cudaFilterModeLinear;
	inputProcessed.normalized = false;

	dim3 dimBlock(FS_BLOCKSIZE_X,FS_BLOCKSIZE_Y);
	dim3 dimGrid((width+dimBlock.x-1)/dimBlock.x, (height+dimBlock.y-1)/dimBlock.y);
	GaussianBlurHorizontalKernel<<<dimGrid, dimBlock>>>(tmpGauss, GAUSSIAN_WINDOW_RADIUS, width, height);
	}
	
	{
        if(cudaBindTexture2D(0, inputProcessed, tmpGauss, channelDesc, width, height, width*sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Cannot bind intermediate blur texture!");
		return;
	}
        inputProcessed.addressMode[0] = cudaAddressModeClamp;
        inputProcessed.addressMode[1] = cudaAddressModeClamp;
        inputProcessed.filterMode = cudaFilterModeLinear;
	inputProcessed.normalized = false;

	dim3 dimBlock(FS_BLOCKSIZE_X,FS_BLOCKSIZE_Y);
	dim3 dimGrid((width+dimBlock.x-1)/dimBlock.x, (height+dimBlock.y-1)/dimBlock.y);
	GaussianBlurVerticalKernel<<<dimGrid, dimBlock>>>(tmpGray, GAUSSIAN_WINDOW_RADIUS, width, height);
	}
	
        if(cudaBindTexture2D(0, inputProcessed, tmpGray, channelDesc, width, height, width*sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Cannot bind preprocessed input texture!");
		return;
	}
        inputProcessed.addressMode[0] = cudaAddressModeClamp;
        inputProcessed.addressMode[1] = cudaAddressModeClamp;
        inputProcessed.filterMode = cudaFilterModeLinear;
	inputProcessed.normalized = false;
}

/*	Function: segmentAndAddToBackground
 * 	-----------------------------------
 * 	This function segments the current input image using the current background model and, afterwards,
 * 	updates the background model with the current image using the learning rate.
 * 
 * 	This function assumes that the input image is already bound to the inputProcessed texture!
 */
void segmentAndAddToBackground(float* segmented, float* background, unsigned width, unsigned height, float rate) {
	// Bind background texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        if(cudaBindTexture2D(0, background_tex, background, channelDesc, width, height, width*sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Cannot bind background texture!");
		return;
	}
        background_tex.addressMode[0] = cudaAddressModeClamp;
        background_tex.addressMode[1] = cudaAddressModeClamp;
        background_tex.filterMode = cudaFilterModeLinear;
	background_tex.normalized = false;
	
	dim3 dimBlock(FS_BLOCKSIZE_X, FS_BLOCKSIZE_Y);
	dim3 dimGrid((width+dimBlock.x-1)/dimBlock.x, (height+dimBlock.y-1)/dimBlock.y);
	SegmentAndUpdateBackgroundKernel<<<dimGrid, dimBlock>>>(segmented, background, width, height, rate);
}