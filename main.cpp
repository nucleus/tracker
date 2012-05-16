#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>
#include <utility>
#include <getopt.h>

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
					 "-a <detection algorithm>, [0-motion estimation, 1-clustering, 2-generalized hough transform]\n";

int main(int argc, char** argv) {
	int opt, tmp;
	
	string parse, sInfile = "", sOutfile = "";
	
	// Processing flags
	bool bSourceIsFile = true;
	bool bDestIsFile = true;
	unsigned int iModelFrames = DEFAULT_MODEL_TRAINING_COUNT;
	double dLearningRate = DEFAULT_LEARNING_RATE;
	detectionAlgorithm algo = ALGO_OPTICAL;
	
	// Source video information
	unsigned int iInputWidth, iInputHeight, iInputFps;
	
	while((opt = getopt(argc, argv, "s:i:d:o:f:l:a:")) != -1) {
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
	
	// Build initial bg model
	for(int i = 0; i < iModelFrames; i++) {
		Mat frame;
		cap >> frame;
		fg.addFrameToModel(frame);
	}
	
	// For debug, write-back of the bg model images
	imwrite("bgmodel.jpg", fg.modelMean());
	
	Mat dst(iInputHeight, iInputWidth, CV_8UC3);
	Mat result(iInputHeight, iInputWidth, CV_8UC3);
	
	// Main tracking loop
	Mat frame, segmentedFrame, previousFrame;
	bool updateBackground = true;
	bool firstFrame = true;
	
	while(1) {
		vector< pair<unsigned,unsigned> > cForegroundList;
		cap >> frame;
		
		if(updateBackground)
			fg.addFrameToModel(frame);
		updateBackground = !updateBackground;
// 		cout << "Updated background model!" << endl;
		
		fg.segment(frame, segmentedFrame, cForegroundList);
		// cout << "Segmented frame!" << endl;

		if(!firstFrame)
			bd.searchBall(segmentedFrame, frame, cForegroundList, algo);
		firstFrame = false;
// 		cout << "Located ball!" << endl;
		
		vb << frame;
		if(!bDestIsFile)
			if(waitKey(30) >= 0) break;
			
		previousFrame = segmentedFrame.clone();
		bd.updatePreviousImage(&previousFrame);
	}

	return 0;
}


