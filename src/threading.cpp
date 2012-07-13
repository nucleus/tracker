/*
 * 	File: threading.cpp
 * 	-------------------
 * 	Implementations for CPU threading.
 * 
 * 	Written by Michael Andersch, 2012.
 */

#include <vector>
#include <utility>

#include "ForegroundSegmenter.h"
#include "BallDetection.h"
#include "VideoBackend.h"
#include "threading.h"

#include <boost/thread.hpp>

using namespace boost;
using namespace boost::this_thread;

ReaderThread::ReaderThread(VideoCapture* _cap, SynchronisedQueue<Mat*>* queue, unsigned w, unsigned h) {
	cap = _cap;
	Q = queue;
	width = w;
	height = h;
}

void ReaderThread::operator()() {
	while(true) {
		frame = new Mat(height, width, CV_8UC3);
		(*cap) >> (*frame);
		Q->Enqueue(frame);
// 		std::cout << "Producer: Enqueued " << frame << std::endl;
	}
}

ProcessorThread::ProcessorThread(SynchronisedQueue<Mat*>* input, SynchronisedQueue<Mat*>* output, bool useGPU, unsigned width, unsigned height, unsigned maxframes, double lrate, detectionAlgorithm _algo) {
	InputQueue = input;
	OutputQueue = output;
	
	bUseGPU = useGPU;
	
	fg.setImageParams(width, height, 1);
	fg.setMaxFrames(maxframes);
	fg.setLearningRate(lrate);
	fg.useGPU(useGPU);
	
	bd.setImageParams(width, height, 1);
	
	updateBackground = true;
	firstFrame = true;
	
	algo = _algo;
	
	processedFrames = 0;
	maxFrames = maxframes;
}
	
void ProcessorThread::operator()() {
	Mat segmentedFrame, previousImage;
	
	for(int i = 0; i < maxFrames; i++) {
		Mat* in = InputQueue->Dequeue();
		fg.addFrameToModel(*in);
		delete in;
	}
	
	while(true) {
		Mat* in = InputQueue->Dequeue();
		vector< pair<unsigned, unsigned> > fglist;
// 		std::cout << "Processor: Received data " << in << std::endl;
// 		if(bUseGPU)
// 			fg.uploadPreprocessFrame(*in);
// 		if(updateBackground) {
// 			fg.addFrameToModel(*in);
// 		}
// 		updateBackground = !updateBackground;
// 		
// 		fg.segment(*in, segmentedFrame, fglist);
// 		std::cout << "Foreground list length: " << fglist.size() << std::endl;
		/* WRAPPER CODE */
// 		if(bUseGPU) {
// 			for(int y = 0; y < frame.rows; y++)
// 				for(int x = 0; x < frame.cols; x++)
// 					if(segmentedFrame.at<float>(y,x) == 255.0)
// 						fglist.push_back(make_pair(x,y));		
// 		}
		/* WRAPPER CODE */
		
// 		if(!firstFrame)
// 			bd.searchBall(segmentedFrame, *in, fglist, algo);
// 		firstFrame = false;
		OutputQueue->Enqueue(in);
// 		std::cout << "Processor: Enqueued data " << in << std::endl;
	}
}
