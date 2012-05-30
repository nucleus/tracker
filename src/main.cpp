#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>
#include <utility>
#include <getopt.h>

#include <sys/time.h>
typedef struct timeval timer;

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "global.h"
#include "ForegroundSegmenter.h"
#include "BallDetection.h"
#include "VideoBackend.h"

using namespace std;
using namespace cv;

// Templated to-string conversion function
template<typename T>
string convert(T sym) {
	stringstream ss;
	ss << sym;
	return ss.str();
}

const string usage = "Usage: ./tracker [options]\n\t"
					 "-s <input device> [CAM/DISK]\n\t"
					 "-i <input video file>\n\t"
					 "-d <output device> [SCREEN/DISK]\n\t"
					 "-o <output video file>\n\t"
					 "-f <background model frames>\n\t"
					 "-l <learning rate>, interval: [0,1]\n\t"
					 "-a <detection algorithm>, [0-motion estimation, 1-clustering, 2-generalized hough transform]\n\t"
					 "-g [use GPU for processing if specified]\n";

// Timing function usable for profiling
double timevaldiff(timer& prior, timer& latter) {
	double x =
	(double)(latter.tv_usec - prior.tv_usec) / 1000.0 +
	(double)(latter.tv_sec - prior.tv_sec) * 1000.0;
	return x;
}

int main(int argc, char** argv) {
	int opt, tmp;
	
	string parse, sInfile = "", sOutfile = "";
	
	// Processing flags
	bool bSourceIsFile = true;
	bool bDestIsFile = true;
	bool bUseGPU = false;
	unsigned int iModelFrames = DEFAULT_MODEL_TRAINING_COUNT;
	double dLearningRate = DEFAULT_LEARNING_RATE;
	detectionAlgorithm algo = ALGO_OPTICAL;
	
	// Source video information
	unsigned int iInputWidth, iInputHeight, iInputFps;
	
	while((opt = getopt(argc, argv, "s:i:d:o:f:l:a:g")) != -1) {
		switch(opt) {
			case 's':
				parse.assign(optarg);
				if(parse == "CAM")
					bSourceIsFile = false;
				else if(parse == "DISK") {}
				else {
					cerr << ERROR("unrecognized video source") << endl << usage;
					return EXIT_FAILURE;
				}
				break;
			case 'i':
				sInfile.assign(optarg);
				break;
			case 'd':
				parse.assign(optarg);
				if(parse == "SCREEN")
					bDestIsFile = false;
				else if(parse == "DISK") {}
				else {
					cerr << ERROR("unrecognized output destination") << endl << usage;
					return EXIT_FAILURE;
				}
				break;
			case 'o':
				sOutfile.assign(optarg);
				break;
			case 'f':
				iModelFrames = atoi(optarg);
				if(iModelFrames < 1 || iModelFrames > 10000) {
					cerr << ERROR("bad model frame count") << endl;
					return EXIT_FAILURE;
				}
				break;
			case 'l':
				dLearningRate = atof(optarg);
				if(dLearningRate <= 0.0 || dLearningRate >= 1.0) {
					cerr << ERROR("bad model learning rate") << endl;
					return EXIT_FAILURE;
				}
				break;
			case 'a':
				tmp = atoi(optarg);
				if(tmp == 0)
					algo = ALGO_OPTICAL;
				else if(tmp == 1)
					algo = ALGO_CLUSTER;
				else if(tmp == 2)
					algo = ALGO_MOVING;
				else {
					cerr << usage;
					return EXIT_FAILURE;
				}
				break;
			case 'g':
				bUseGPU = true;
				break;
			default:
				cerr << usage;
				return EXIT_FAILURE;
				break;
		}
	}
	
	if((bSourceIsFile && sInfile == "") || bDestIsFile && sOutfile == "") {
		cerr << ERROR("missing file specification") << endl << usage;
		return EXIT_FAILURE;
	}
	
	/** FRONT END SETUP BEGIN **/
	VideoCapture cap;	
	// Try to open the video stream
	if(bSourceIsFile) {
		cap.open(sInfile);
	} else {
		cap.open(DEFAULT_CAMERA);
	}
	
	if(!cap.isOpened()) {
		cerr << ERROR("could not open video source");
		return EXIT_FAILURE;
	}
	
	iInputWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	iInputHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	if(bSourceIsFile) {
		iInputFps = 120; //cap.get(CV_CAP_PROP_FPS) * 2 + 1;
	} else { // Framerate get impossible for webcam devices
		iInputFps = 25;
	}
	cout << "VIDEO SOURCE: " << iInputWidth << "x" << iInputHeight << " @ " << iInputFps << "Hz" << endl;
	/** FRONT END SETUP END **/
	
	/** BACK END SETUP BEGIN **/
	VideoBackend vb(bDestIsFile ? "DISK" : "SCREEN");
	if(bDestIsFile)
		vb.setFileParams(sOutfile, CV_FOURCC('D','I','V','X'), iInputFps, Size(iInputWidth, iInputHeight), true);
	cout << "VIDEO DESTINATION: " << sOutfile << " @ DIVX codec" << endl;
	/** BACK END SETUP END **/
	
	BallDetection bd;
	bd.setImageParams(iInputWidth, iInputHeight, 1);
	
	ForegroundSegmenter fg;
	fg.setImageParams(iInputWidth, iInputHeight, 1);
	fg.setMaxFrames(iModelFrames);
	fg.setLearningRate(dLearningRate);
	fg.useGPU(bUseGPU);
	
	// If CPU background modeling is used, train initial frames to model
	if(!bUseGPU) {		
		for(int i = 0; i < iModelFrames; i++) {
			Mat frame;
			cap >> frame;
			fg.addFrameToModel(frame);
		}
	}
	
	Mat dst(iInputHeight, iInputWidth, CV_8UC3);
	Mat result(iInputHeight, iInputWidth, CV_8UC3);
	
	// Main tracking loop
	Mat frame, segmentedFrame, previousImage;
	bool updateBackground = true;
	bool firstFrame = true;
	double framerate_avg = INITIAL_FPS;
	unsigned long long frames_processed = 1;
	
	timer start, end;
	
	while(1) {
		vector< pair<unsigned,unsigned> > cForegroundList;
		
		// Capture current frame and measure time
		gettimeofday(&start, NULL);
		cap >> frame;
		gettimeofday(&end, NULL);
		double capture = (double)timevaldiff(start,end);
		
		// In case of GPU processing, upload and prepare frame
		gettimeofday(&start, NULL);
		if(bUseGPU) {
			fg.uploadPreprocessFrame(frame);
		}
		gettimeofday(&end, NULL);
		double preprocess = (double)timevaldiff(start,end);
		
		// Update background model for every second image and measure time
		gettimeofday(&start, NULL);
		if(updateBackground) {
			fg.addFrameToModel(frame);
		}
		updateBackground = !updateBackground;
		gettimeofday(&end, NULL);
		double backgroundadd = (double)timevaldiff(start, end);
		
		// Segment foreground for current frame and measure time
		gettimeofday(&start, NULL);
		fg.segment(frame, segmentedFrame, cForegroundList);
		gettimeofday(&end, NULL);
		double segmentation = (double)timevaldiff(start, end);
		
		/** THIS IS WRAPPER CODE, REMOVE ASAP */
		if(bUseGPU) {
			for(int y = 0; y < frame.rows; y++)
				for(int x = 0; x < frame.cols; x++)
					if(segmentedFrame.at<float>(y,x) == 255.0)
						cForegroundList.push_back(make_pair(x,y));		
		}
		/** THIS IS WRAPPER CODE, REMOVE ASAP */
		
		// Detect ball among foreground objects and measure time
		gettimeofday(&start, NULL);
		if(!firstFrame)
			bd.searchBall(segmentedFrame, frame, cForegroundList, algo);
		firstFrame = false;
		gettimeofday(&end, NULL);
		double detection = (double)timevaldiff(start,end);
		
// 		cout << "Preprocess: " << preprocess << "ms, Capture: " << capture << "ms, Segmentation: "<< segmentation << "ms, Detection: "<< detection << "ms" << endl;
		
		// Calculate maximum sustainable frame rate
		double total = preprocess + backgroundadd + segmentation + detection;// + capture;
		double maxfps = 1.0/(total/1000.0);
		if(maxfps < 1.0)
			maxfps = 1.0;
		
		// Calculate momentary throughput and latency stats
		framerate_avg *= frames_processed;
		framerate_avg += maxfps;
		frames_processed++;
		framerate_avg /= frames_processed;
 		cout << "Instantaneous frame rate: " << maxfps << ", average frame rate: " << framerate_avg << "\r" << flush;
 		putText(frame, convert(framerate_avg), Point(10,30), FONT_HERSHEY_PLAIN, 1, Scalar::all(255), 2, 8);
		
		// Output frame to disk / screen
		vb << frame;
		if(!bDestIsFile)
			if(waitKey(30) >= 0) break;

		// For optical flow based ball detection: update motion reference frame
		previousImage = segmentedFrame.clone();
		bd.updatePreviousImage(&previousImage);
	}

	return 0;
}


