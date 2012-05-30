/*
 *	File: ForegroundSegmenter.h
 *	---------------------------
 *	The class definition of the foreground segmentation engine.
 *
 *	Author: Michael Andersch, 2012
 */

#ifndef FOREGROUND_SEG_H_
#define FOREGROUND_SEG_H_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cassert>
#include <vector>
#include <utility>

#include <cuda.h>
#include <cutil_inline.h>

#include "global.h"

using namespace cv;
using namespace std;

// CUDA reference functions
void createKernel1D(unsigned ksize, string type);
void segmentAndAddToBackground(float* segmented, float* background, unsigned width, unsigned height, float rate);
void preProcessImage(uchar4* src, float* tmpGray, float* tmpGauss, float* dst, unsigned width, unsigned height);

// Class containing the background model and foreground segmentation functionality
class ForegroundSegmenter {
public:
	ForegroundSegmenter();	
	~ForegroundSegmenter();
	
	/*	Function: setImageParams
	 *	------------------------
	 *	Specify the core parameters of the background model image. 
	 */
	void setImageParams(unsigned int _width, unsigned int _height, unsigned int _channels);
	
	/*	Function: uploadPreprocessFrame
	 * 	-------------------------------
	 * 	If the GPU is used for background removal, this function uploads the current frame to the device,
	 * 	converts it to grayscale and performs gaussian smoothing. 
	 */
	void uploadPreprocessFrame(Mat& frame);
	
	/*	Function: addFrameToModel
	 *	-------------------------
	 *	Adds an incoming frame to the current background model.
	 *	If the number of modeled frames is below the specified maximum,
	 *	each frame is weighted with 1/MAXFRAMES for initial bg model training.
	 *	After initial training is over, frames are added with a weight given
	 *	by the learning rate.
	 */
	void addFrameToModel(Mat& frame);
	
	/*	Function: segment
	 *	-----------------
	 *	Utilizes the available background model to remove the background from
	 *	a given source frame, storing the result into the dstFrame location.
	 */
	void segment(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList);
	
	/*	Function: modelMean
	 *	-------------------
	 *	Returns the mean image mu of the background model.
	 */
	Mat& modelMean() {
		return cBgMean;
	}
	
	/*	Function: useEdgeImages
	 *	-------------------
	 *	Specify whether the background model will be an edge image model.
	 */
	void useEdgeImages(bool b = true) {
		assert(iCurrentFramesModeled == 0);
		bProcessEdgeImages = b;
	}
	
	/*	Function: useGPU
	 *	-------------------
	 *	Specify whether the background model will be computed on the GPU.
	 */
	void useGPU(bool b = true);
	
	/*	Function: setMaxFrames
	 *	-------------------
	 *	Specify the maximum number of background frames to be included in the bg model.
	 */
	void setMaxFrames(unsigned int max) {
		assert(iCurrentFramesModeled <= max);
		iMaxFramesModeled = max;
	}
	
	/*	Function: setLearningRate
	 *	-------------------------
	 *	Specify the weight at which (after initial training) new frames are added to the model
	 */
	void setLearningRate(double alpha) {
		dLearningRate = alpha;
	}
	
	/*	Function: reset
	 *	-------------------
	 *	Deletes the current background model.
	 */
	void reset() {
		iCurrentFramesModeled = 0;
	}
private:
	// Current background model (CPU)
	Mat cBgMean;
	// Processing flags
	bool bProcessEdgeImages;
	bool bProcessOnGPU;
	// Frame information
	unsigned int iCurrentFramesModeled;
	unsigned int iMaxFramesModeled;
	unsigned int iBgWidth, iBgHeight, iBgChannels;
	double dLearningRate;
	// GPU data storage
	uchar4* h_frame, *d_frame;
	float* h_dst, *d_dst, *d_tmpGray, *d_tmpGauss, *d_background;
};

#endif