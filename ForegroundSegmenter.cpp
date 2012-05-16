/*
 *	File: ForegroundSegmenter.cpp
 *	---------------------------
 *	The function implementations of the foreground segmentation engine.
 *
 *	Author: Michael Andersch, 2012
 */

#include "ForegroundSegmenter.h"

ForegroundSegmenter::ForegroundSegmenter() : iCurrentFramesModeled(0), iMaxFramesModeled(0), bProcessEdgeImages(false), dLearningRate(DEFAULT_LEARNING_RATE) {};

void ForegroundSegmenter::setImageParams(unsigned int _width, unsigned int _height, unsigned int _channels) {
	assert(iCurrentFramesModeled == 0);
	iBgWidth = _width;
	iBgHeight = _height;
	assert(_channels == 1);
	iBgChannels = _channels;
	
// 	namedWindow("RunningBGAverage", 1);
	
	cBgMean = Mat(iBgHeight, iBgWidth, CV_32FC1, Scalar(0.0)).clone();
	cBgDeviation = Mat(iBgHeight, iBgWidth, CV_32FC1, Scalar(0.0)).clone();
}

void ForegroundSegmenter::addFrameToModel(Mat& frame) {
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
		Mat tmp;
		cBgMean.convertTo(tmp, CV_8UC1);
// 		imshow("RunningBGAverage", tmp);
	}
}

void ForegroundSegmenter::segment(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned,unsigned> >& cForegroundList)  {
	Mat tmp;
	Vec3b zeros = {0,0,0};
	cvtColor(srcFrame, tmp, CV_RGB2GRAY);
	GaussianBlur(tmp, tmp, Size(GAUSSIAN_WINDOW_RADIUS,GAUSSIAN_WINDOW_RADIUS), 2, 2);
	tmp.convertTo(tmp, CV_32FC1);

	absdiff(tmp, cBgMean, tmp);
	multiply(tmp, tmp, tmp);
	
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
}