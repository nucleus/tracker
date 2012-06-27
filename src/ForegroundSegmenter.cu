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

#if (__CUDA_ARCH__ < 200)
#define IMAD(a,b,c) (__mul24((a),(b)) + (c))
#else
#define IMAD(a,b,c) ((a)*(b)+(c))
#endif

texture<uchar4, 2, cudaReadModeElementType> input;		// input image, unmodified
texture<float, 2, cudaReadModeElementType> inputProcessed;	// input image, grayscale and blurred
texture<float, 2, cudaReadModeElementType> background_tex;	// current background image
texture<float, 2, cudaReadModeElementType> binary_candidates;	// binary background mask

__device__ __constant__ float kernel[GAUSSIAN_WINDOW_RADIUS];
__device__ __constant__ float kernel_wide[GAUSSIAN_WIDTH_CAND];

static int firstFrameProcessed = 1;

/*	Function: NonMaximumSuppressionKernel
 * 	------------------------------------------
 * 	Processes the strongly blurred binary foreground image to perform
 *	non-maximum suppression on windows of pels. Maximum pixels are re-
 *	ported as possible ball candidates. 
 */
__global__ void NonMaximumSuppressionKernel(uint32_t* candidates, unsigned width, unsigned height) {
	unsigned x = IMAD(blockIdx.x, blockDim.x, threadIdx.x);
	unsigned y = IMAD(blockIdx.y, blockDim.y, threadIdx.y);
	
	if(x >= width || y >= height)
		return;
	
	float sample = tex2D(binary_candidates, x+0.5f, y+0.5f);

	// If background pixel, return
	if(sample < 1.0f)
		return;
	
	// Non-maximum suppression
	float max = sample;
	for(int i = -GAUSSIAN_WIDTH_CAND/2; i <= GAUSSIAN_WIDTH_CAND/2; i++) {
		for(int j = -GAUSSIAN_WIDTH_CAND/2; j <= GAUSSIAN_WIDTH_CAND/2; j++) {
			max = fmax(max, tex2D(binary_candidates, x+j+0.5f, y+i+0.5f));
		}
	}
	
	// Generate the candidate array
	if(max == sample) {
		candidates[atomicAdd(candidates, 1)+1] = PACKU32(x,y);
	}
}

/*	Function: SegmentAndUpdateBackgroundKernel
 * 	------------------------------------------
 * 	Segments the preprocessed input image using the background model.
 * 	Output is a foreground binary image and an updated background model. 
 */
__global__ void SegmentAndUpdateBackgroundKernel(float* segmented, float* background, unsigned width, unsigned height, float rate) {
	unsigned x = IMAD(blockIdx.x, blockDim.x, threadIdx.x);
	unsigned y = IMAD(blockIdx.y, blockDim.y, threadIdx.y);
	unsigned offset = IMAD(y, width, x);
	
	if(x >= width || y >= height)
		return;
	
	float in_sample = tex2D(inputProcessed, x+0.5f, y+0.5f);
	float last_bg_sample = tex2D(background_tex, x+0.5f, y+0.5f);
	float new_bg_sample = last_bg_sample * (1.0f-rate) + in_sample * rate;
	float binary;
	
	in_sample = fabs(in_sample - last_bg_sample);
	in_sample = in_sample * in_sample;

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
	unsigned x = IMAD(blockIdx.x, blockDim.x, threadIdx.x);
	unsigned y = IMAD(blockIdx.y, blockDim.y, threadIdx.y);
	unsigned offset = IMAD(y, width, x);
	
	if(x >= width || y >= height)
		return;
	
	float sum = 0;
#pragma unroll
	for(int i = 0; i < ksize; i++)
		sum += kernel[i] * tex2D(inputProcessed, x-ksize/2+i+0.5f, y+0.5f);
	dst[offset] = sum;
}	

/*	Function: GaussianBlurVerticalKernel
 * 	------------------------------------
 * 	The vertical pass of the gaussian filtering operation.
 */
__global__ void GaussianBlurVerticalKernel(float* dst, unsigned ksize, unsigned width, unsigned height) {
	unsigned x = IMAD(blockIdx.x, blockDim.x, threadIdx.x);
	unsigned y = IMAD(blockIdx.y, blockDim.y, threadIdx.y);
	unsigned offset = IMAD(y, width, x);
	
	if(x >= width || y >= height)
		return;
	
	float sum = 0;
#pragma unroll
	for(int i = 0; i < ksize; i++)
		sum += kernel[i] * tex2D(inputProcessed, x-ksize/2+i+0.5f, y+0.5f);
	dst[offset] = sum;
}

/*	Function: WideGaussianBlurHorizontalKernel
 * 	------------------------------------------
 * 	The horizontal pass of the wide gaussian filtering operation.
 */
__global__ void WideGaussianBlurHorizontalKernel(float* dst, unsigned ksize, unsigned width, unsigned height) {
	unsigned x = IMAD(blockIdx.x, blockDim.x, threadIdx.x);
	unsigned y = IMAD(blockIdx.y, blockDim.y, threadIdx.y);
	unsigned offset = IMAD(y, width, x);
	
	if(x >= width || y >= height)
		return;
	
	float sum = 0;
#pragma unroll
	for(int i = 0; i < ksize; i++)
		sum += kernel_wide[i] * tex2D(inputProcessed, x-ksize/2+i+0.5f, y+0.5f);
	dst[offset] = sum;
}	

/*	Function: WideGaussianBlurVerticalKernel
 * 	----------------------------------------
 * 	The vertical pass of the wide gaussian filtering operation.
 */
__global__ void WideGaussianBlurVerticalKernel(float* dst, unsigned ksize, unsigned width, unsigned height) {
	unsigned x = IMAD(blockIdx.x, blockDim.x, threadIdx.x);
	unsigned y = IMAD(blockIdx.y, blockDim.y, threadIdx.y);
	unsigned offset = IMAD(y, width, x);
	
	if(x >= width || y >= height)
		return;
	
	float sum = 0;
#pragma unroll
	for(int i = 0; i < ksize; i++)
		sum += kernel_wide[i] * tex2D(inputProcessed, x-ksize/2+i+0.5f, y+0.5f);
	dst[offset] = sum;
}

/*	Function: Rgb2GrayKernel
 * 	------------------------
 * 	This kernel converts the input image in texture memory to grayscale and stores it to dst.
 */
__global__ void Rgb2GrayKernel(float *dst, unsigned width, unsigned height) {
	unsigned x = IMAD(blockIdx.x, blockDim.x, threadIdx.x);
	unsigned y = IMAD(blockIdx.y, blockDim.y, threadIdx.y);
	unsigned offset = IMAD(y, width, x);
	
	if(x >= width || y >= height)
		return;

	uchar4 sample;
	sample = tex2D(input, x+0.5f, y+0.5f);
	float gray = sample.x * 0.2989f + 0.587f * sample.y + 0.114f * sample.z;
	dst[offset] = gray;
}

/*	Function: CoalescedRgb2GrayKernel
 * 	---------------------------------
 * 	This kernel converts the input image in texture memory to grayscale and stores it to dst.
 * 
 * 	The difference to the kernel above is the direct reading of the OpenCV Vec3b input
 * 	image into the kernel. To ensure coalesced accesses, shared memory is used.
 */
__global__ void CoalescedRgb2GrayKernel(uchar* src, float* dst, unsigned length) {
	unsigned idx = IMAD(3*blockIdx.x, blockDim.x, threadIdx.x);
	unsigned idx_store = IMAD(blockIdx.x, blockDim.x, threadIdx.x);
	
	__shared__ uchar dataTile[3*RGB2GRAY_BS_X];
	
	if(idx < 3*length)
		dataTile[threadIdx.x] = src[idx];
	if(idx+RGB2GRAY_BS_X < 3*length)
		dataTile[threadIdx.x+RGB2GRAY_BS_X] = src[idx+RGB2GRAY_BS_X];
	if(idx+2*RGB2GRAY_BS_X < 3*length)
		dataTile[threadIdx.x+2*RGB2GRAY_BS_X] = src[idx+2*RGB2GRAY_BS_X];
	__syncthreads();
	uchar3 sample = ((uchar3*)dataTile)[threadIdx.x];
	
	float gray = sample.x * 0.114f + 0.587f * sample.y + 0.2989f * sample.z;
	
	if(idx_store < length)
		dst[idx_store] = gray;
}

/*	Function: createKernel1D
 * 	------------------------
 * 	This function generates 1-dimensional gaussian kernels for preprocessing.
 */
void createKernel1D(unsigned ksize, string type) {
	if(ksize%2 != 1) {
		fprintf(stderr, "Bad kernel size in kernel creation, aborting");
		exit(EXIT_FAILURE);
	}
	
	float kern[ksize];
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

/*	Function: createCandidateKernel1D
 * 	---------------------------------
 *	This function generates 1-dimensional gaussian kernels for candidate detection. 
 */
void createCandidateKernel1D(unsigned ksize, string type) {
	if(ksize%2 != 1) {
		fprintf(stderr, "Bad kernel size in kernel creation, aborting");
		exit(EXIT_FAILURE);
	}
	
	float kern[ksize];
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
	cudaMemcpyToSymbol("kernel_wide", kern, sizeof(kern));
}

/*	Function: preProcessImage
 * 	-------------------------
 * 	This function operates on an RGBA color image in the device. It will first launch a kernel to perform
 * 	RGBA to grayscale conversion (which also converts uchar4 to float). Then, a gaussian blur filter is
 * 	implemented using separation into horizontal and vertical passes. 
 */
void preProcessImage(uchar3* src, float* tmpGray, float* tmpGauss, float* background, unsigned width, unsigned height) {
#if 0
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
#endif
	{
	unsigned length = width * height;
	dim3 dimBlock(RGB2GRAY_BS_X, 1);
	dim3 dimGrid((length+dimBlock.x-1)/dimBlock.x, 1);
	CoalescedRgb2GrayKernel<<<dimGrid, dimBlock>>>((uchar*)src, tmpGray, length);
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
	
	// If this is the first frame ever processed, initialize background model to get the computation going
	if(firstFrameProcessed) {
		cutilSafeCall( cudaMemcpy(background, tmpGray, width*height*sizeof(float), cudaMemcpyDeviceToDevice) );
		firstFrameProcessed = 0;
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

/*	Function: calculateLowLevelCandidates
 * 	-------------------------------------
 * 	This function is the GPU implementation of low-level ball candidate generation.
 * 	First, a wide gaussian filtering of the binary background mask is performed, which is
 * 	followed by non-maximum suppression. Remaining pixels are reported as possible ball candidates.
 */
void calculateLowLevelCandidates(float* binary, float* temporary, float* h_dbg, uint32_t* candidates, unsigned width, unsigned height) {
// 	cudaEvent_t start, end;
// 	cudaEventCreate(&start);
// 	cudaEventCreate(&end);
// 	cudaEventRecord(start, 0);
	
	cudaMemsetAsync(candidates, 0, sizeof(uint32_t));
	dim3 dimBlock(FS_BLOCKSIZE_X, FS_BLOCKSIZE_Y);
	dim3 dimGrid((width+dimBlock.x-1)/dimBlock.x, (height+dimBlock.y-1)/dimBlock.y);
	
	{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	if(cudaBindTexture2D(0, inputProcessed, binary, channelDesc, width, height, width*sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Cannot bind wide gaussian input as texture!");
		return;
	}
        inputProcessed.addressMode[0] = cudaAddressModeClamp;
        inputProcessed.addressMode[1] = cudaAddressModeClamp;
        inputProcessed.filterMode = cudaFilterModeLinear;
	inputProcessed.normalized = false;
	}
	WideGaussianBlurHorizontalKernel<<<dimGrid, dimBlock>>>(temporary, GAUSSIAN_WIDTH_CAND, width, height);
	
	{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	if(cudaBindTexture2D(0, inputProcessed, temporary, channelDesc, width, height, width*sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Cannot bind wide gaussian intermediate as texture!");
		return;
	}
        inputProcessed.addressMode[0] = cudaAddressModeClamp;
        inputProcessed.addressMode[1] = cudaAddressModeClamp;
        inputProcessed.filterMode = cudaFilterModeLinear;
	inputProcessed.normalized = false;
	}
	WideGaussianBlurVerticalKernel<<<dimGrid, dimBlock>>>(binary, GAUSSIAN_WIDTH_CAND, width, height);
	
	{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	if(cudaBindTexture2D(0, binary_candidates, binary, channelDesc, width, height, width*sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Cannot bind binary mask as texture!");
		return;
	}
        inputProcessed.addressMode[0] = cudaAddressModeClamp;
        inputProcessed.addressMode[1] = cudaAddressModeClamp;
        inputProcessed.filterMode = cudaFilterModeLinear;
	inputProcessed.normalized = false;
	}
	NonMaximumSuppressionKernel<<<dimGrid, dimBlock>>>(candidates, width, height);
// 	cudaEventRecord(end, 0);
// 	cudaEventSynchronize(end);
// 	float elapsed;
// 	cudaEventElapsedTime(&elapsed, start, end);
// 	fprintf(stderr, "Time taken for candidate generation: %3.3f ms", elapsed);
// 	cudaEventDestroy(start);
// 	cudaEventDestroy(end);
}