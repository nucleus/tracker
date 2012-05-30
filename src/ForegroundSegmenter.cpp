/*
 *	File: ForegroundSegmenter.cpp
 *	---------------------------
 *	The function implementations of the foreground segmentation engine.
 *
 *	Author: Michael Andersch, 2012
 */

#include "ForegroundSegmenter.h"

ForegroundSegmenter::ForegroundSegmenter() : iCurrentFramesModeled(0), iMaxFramesModeled(0), bProcessEdgeImages(false), dLearningRate(DEFAULT_LEARNING_RATE) {};

ForegroundSegmenter::~ForegroundSegmenter() {
	if(bProcessOnGPU) {
		// In-device memories
		cudaFree(d_frame);
		cudaFree(d_dst);
		cudaFree(d_tmpGray);
		cudaFree(d_tmpGauss);
		cudaFree(d_background);
		// Page-locked host memories
		cudaFreeHost(h_frame);
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
	if(cuInit(0) != CUDA_SUCCESS) {
		exit(EXIT_FAILURE);
	}
	
	// If GPU is used, allocate memories for incoming frames and bg model
	cutilSafeCall( cudaHostAlloc((void**)&h_frame, iBgWidth*iBgHeight*sizeof(uchar4), cudaHostAllocWriteCombined) );
	cutilSafeCall( cudaHostAlloc((void**)&h_dst, iBgWidth*iBgHeight*sizeof(float), cudaHostAllocDefault) );
	cutilSafeCall( cudaMalloc((void**)&d_frame, iBgWidth*iBgHeight*sizeof(uchar4)) );
	cutilSafeCall( cudaMalloc((void**)&d_dst, iBgWidth*iBgHeight*sizeof(float)) );
	cutilSafeCall( cudaMalloc((void**)&d_tmpGray, iBgWidth*iBgHeight*sizeof(float)) );
	cutilSafeCall( cudaMalloc((void**)&d_tmpGauss, iBgWidth*iBgHeight*sizeof(float)) );
	cutilSafeCall( cudaMalloc((void**)&d_background, iBgWidth*iBgHeight*sizeof(float)) );
	
	// Generate the smoothing kernel for the GPU
	createKernel1D(GAUSSIAN_WINDOW_RADIUS, "gaussian");
}

void ForegroundSegmenter::uploadPreprocessFrame(Mat& frame) {
	// Generate frame in uchar4 representation
	Vec3b* walker = frame.ptr<Vec3b>();
	for(int i = 0; i < iBgWidth*iBgHeight; i++)
		h_frame[i] = make_uchar4(walker[i][2], walker[i][1], walker[i][0], 0);
// 	cout << "Modified frame, first sample: " << (int)h_frame[0].x << " " << (int)h_frame[0].y << " " << (int)h_frame[0].z << " " << (int)h_frame[0].w << endl;
	
	// Copy frame to device and perform preprocessing
	cutilSafeCall( cudaMemcpy(d_frame, h_frame, iBgWidth*iBgHeight*sizeof(uchar4), cudaMemcpyHostToDevice) );
	preProcessImage(d_frame, d_tmpGray, d_tmpGauss, d_dst, iBgWidth, iBgHeight);
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
		}
		
		// Model has been initialized, now update
		if(iCurrentFramesModeled == iMaxFramesModeled) {
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
				if(tmp.at<float>(y,x) > THRESH) {
	// 				dstFrame.at<Vec3b>(y,x) = srcFrame.at<Vec3b>(y,x)
					cForegroundList.push_back(make_pair(x,y));
				} else {
	// 				dstFrame.at<Vec3b>(y,x) = zeros;
				}
			}
	} else { // GPU processing
		segmentAndAddToBackground(d_tmpGauss, d_background, iBgWidth, iBgHeight, dLearningRate);
		cutilSafeCall( cudaMemcpy(h_dst, d_tmpGauss, iBgWidth*iBgHeight*sizeof(float), cudaMemcpyDeviceToHost) );
		dstFrame = Mat(iBgHeight, iBgWidth, CV_32FC1, h_dst).clone();
	}
}