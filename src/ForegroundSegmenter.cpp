/*
 *	File: ForegroundSegmenter.cpp
 *	---------------------------
 *	The function implementations of the foreground segmentation engine.
 *
 *	Author: Michael Andersch, 2012
 */

#include "ForegroundSegmenter.h"
#include <stdio.h>
#include <limits.h>
#include <list>

#include <sys/time.h>
typedef struct timeval timer;

ForegroundSegmenter::ForegroundSegmenter() : iCurrentFramesModeled(0), iMaxFramesModeled(0), bProcessEdgeImages(false), dLearningRate(DEFAULT_LEARNING_RATE) {};

ForegroundSegmenter::~ForegroundSegmenter() {
	if(bProcessOnGPU) {
		// In-device memories
		cudaFree(d_frame);
		cudaFree(d_tmpGray);
		cudaFree(d_tmpGauss);
		cudaFree(d_background);
		cudaFree(d_candidates);
		// Page-locked host memories
		cudaFreeHost(h_dst);
	}
}

void ForegroundSegmenter::setImageParams(unsigned int _width, unsigned int _height, unsigned int _channels) {
	assert(iCurrentFramesModeled == 0);
	iBgWidth = _width;
	iBgHeight = _height;
	assert(_channels == 1);
	iBgChannels = _channels;
	
	bProcessEdgeImages = false;
	bProcessOnGPU = false;
	
	cBgMean = Mat(iBgHeight, iBgWidth, CV_32FC1, Scalar(0.0)).clone();
}

void ForegroundSegmenter::useGPU(bool b) {
	assert(iCurrentFramesModeled == 0);
	bProcessOnGPU = b;
	
	// Initialize CUDA
	if(bProcessOnGPU) {
		if(cuInit(0) != CUDA_SUCCESS) {
			exit(EXIT_FAILURE);
		}
		
		// If GPU is used, allocate memories for incoming frames and bg model
		cutilSafeCall( cudaHostAlloc((void**)&h_dst, iBgWidth*iBgHeight*sizeof(float), cudaHostAllocDefault) );
		cutilSafeCall( cudaMalloc((void**)&d_frame, iBgWidth*iBgHeight*sizeof(uchar3)) );
		cutilSafeCall( cudaMalloc((void**)&d_tmpGray, iBgWidth*iBgHeight*sizeof(float)) );
		cutilSafeCall( cudaMalloc((void**)&d_tmpGauss, iBgWidth*iBgHeight*sizeof(float)) );
		cutilSafeCall( cudaMalloc((void**)&d_background, iBgWidth*iBgHeight*sizeof(float)) );
		cutilSafeCall( cudaMalloc((void**)&d_candidates, (ALLOWED_CANDIDATES+1)*sizeof(uint32_t)) );
		cutilSafeCall( cudaMemset(d_candidates, 0, (ALLOWED_CANDIDATES+1)*sizeof(uint32_t)) );
		
		// Generate the smoothing kernels for the GPU
		createKernel1D(GAUSSIAN_WINDOW_RADIUS, "gaussian");
		createCandidateKernel1D(GAUSSIAN_WIDTH_CAND, "gaussian");		
	}
}

void ForegroundSegmenter::uploadPreprocessFrame(Mat& frame) {
	Vec3b* walker = frame.ptr<Vec3b>();
	cutilSafeCall( cudaMemcpy(d_frame, walker, iBgWidth*iBgHeight*sizeof(uchar3), cudaMemcpyHostToDevice) );
	preProcessImage(d_frame, d_tmpGray, d_tmpGauss, d_background, iBgWidth, iBgHeight);
}

void ForegroundSegmenter::addFrameToModel(Mat& frame) {
	// CPU processing
	if(!bProcessOnGPU) {
		Mat f;
		cvtColor(frame, f, CV_RGB2GRAY);
		GaussianBlur(f, f, Size(GAUSSIAN_WINDOW_RADIUS,GAUSSIAN_WINDOW_RADIUS), 2, 2);
		f.convertTo(f, CV_32FC1);
		
		// Add initial frames to model with equal weights
		if(iCurrentFramesModeled < iMaxFramesModeled) {
			iCurrentFramesModeled++;
			addWeighted(cBgMean, 1.0, f, 1.0/(double)iMaxFramesModeled, 0.0, cBgMean);
		}else  { // Model has been initialized, now update
			addWeighted(cBgMean, (1.0-dLearningRate), f, dLearningRate, 0.0, cBgMean);
		}
	} else { // GPU processing
	// If GPU processing is active, frame will be added during background segmentation - do nothing
	}
}

void ForegroundSegmenter::segment(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned,unsigned> >& cForegroundList)  {
	if(!bProcessOnGPU) {
		Mat tmp;
		Vec3b zeros = {0,0,0};
		cvtColor(srcFrame, tmp, CV_RGB2GRAY);
		GaussianBlur(tmp, tmp, Size(GAUSSIAN_WINDOW_RADIUS,GAUSSIAN_WINDOW_RADIUS), 2, 2);
		tmp.convertTo(tmp, CV_32FC1);

		absdiff(tmp, cBgMean, tmp);
		multiply(tmp, tmp, tmp);
		threshold(tmp, tmp, THRESH, 255.0, THRESH_BINARY);
		
		dstFrame = tmp.clone();
		for(int y = 0; y < srcFrame.rows; y++)
			for(int x = 0; x < srcFrame.cols; x++) {
				if(tmp.at<float>(y,x) > THRESH)
					cForegroundList.push_back(make_pair(x,y));
			}
	} else { // GPU processing
		segmentAndAddToBackground(d_tmpGauss, d_background, iBgWidth, iBgHeight, dLearningRate);
		cutilSafeCall( cudaMemcpy(h_dst, d_tmpGauss, iBgWidth*iBgHeight*sizeof(float), cudaMemcpyDeviceToHost) );
		dstFrame = Mat(iBgHeight, iBgWidth, CV_32FC1, h_dst).clone();
	}
}

void ForegroundSegmenter::genLowLevelCandidates(Mat& foreground, uint32_t candidates[ALLOWED_CANDIDATES+1]) {
	if(!bProcessOnGPU) {
		candidates[0] = 0;
		Mat tmp(iBgHeight, iBgWidth, CV_32FC1);
		GaussianBlur(foreground, tmp, Size(GAUSSIAN_WIDTH_CAND,GAUSSIAN_WIDTH_CAND), 2, 2);
		for(int y = 0; y < tmp.rows; y++) {
			for(int x = 0; x < tmp.cols; x++) {
				// for each pixel
				float max = tmp.at<float>(y,x);
				if(max < 1.0)
					continue;
				for(int i = -GAUSSIAN_WIDTH_CAND/2; i <= GAUSSIAN_WIDTH_CAND/2; i++)
					for(int j = -GAUSSIAN_WIDTH_CAND/2; j <= GAUSSIAN_WIDTH_CAND/2; j++) {
						int u = x+j;
						int v = y+i;
						if(u >= 0 && u < tmp.cols && v >= 0 && v < tmp.rows) {
							if(tmp.at<float>(v,u) > max)
								max = tmp.at<float>(v,u);
						}
					}
				if(max == tmp.at<float>(y,x)) {
					candidates[0]++;
					candidates[candidates[0]] = PACKU32(x,y);
				}
			}
		}
	} else {
		calculateLowLevelCandidates(d_tmpGauss, d_tmpGray, h_dst, d_candidates, iBgWidth, iBgHeight);
		cutilSafeCall( cudaMemcpy(candidates, d_candidates, (ALLOWED_CANDIDATES+1)*sizeof(uint32_t), cudaMemcpyDeviceToHost) );	
	}
}