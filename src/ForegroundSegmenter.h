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

// If including both cuda and opencv, this prevents compiler warnings
#undef MIN
#undef MAX

#include <cuda.h>
#include <cutil_inline.h>

#include "global.h"

using namespace cv;
using namespace std;

// CUDA functions
void createKernel1D(unsigned ksize, string type);
void createCandidateKernel1D(unsigned ksize, string type);
void calculateLowLevelCandidates(float* binary, float* temporary, float* h_dbg, uint32_t* candidates, unsigned width, unsigned height);
void segmentAndAddToBackground(float* segmented, float* background, unsigned width, unsigned height, float rate);
void preProcessImage(uchar3* src, float* tmpGray, float* tmpGauss, float* background, unsigned width, unsigned height);
void testFastRgb2Gray(uchar3* src, float* dst, unsigned width, unsigned height);

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
	
	/*	Function: genLowLevelCandidates
	 * 	-------------------------------
	 * 	Processes a given binary foreground mask to generate
	 * 	early ball candidate pixels by strong blur followed by
	 * 	non-maximum suppression. 
	 */
	void genLowLevelCandidates(Mat& foreground, uint32_t candidates[ALLOWED_CANDIDATES+1]);
	
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
	uchar3 *d_frame;
	float *h_dst, *d_tmpGray, *d_tmpGauss, *d_background;
	uint32_t *d_candidates;
};

#endif