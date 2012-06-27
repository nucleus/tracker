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

#include <boost/thread/thread.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "global.h"
#include "ForegroundSegmenter.h"
#include "BallDetection.h"
#include "VideoBackend.h"
#include "threading.h"
#include "OptionParser.h"

using namespace std;
using namespace cv;
using namespace boost;
using namespace boost::this_thread;

// Templated to-string conversion function
template<typename T>
string convert(T sym) {
	stringstream ss;
	ss << sym;
	return ss.str();
}

// Timing function usable for profiling
double timevaldiff(timer& prior, timer& latter) {
	double x =
	(double)(latter.tv_usec - prior.tv_usec) / 1000.0 +
	(double)(latter.tv_sec - prior.tv_sec) * 1000.0;
	return x;
}

int main(int argc, char** argv) {
	OptionParser parser;
	parser.parse(argc, argv, CONFIG_FILE);
	po::variables_map cfg = parser.getOptions();

	string sInfile = "", sOutfile = "";
	
	// Processing flags
	bool bSourceIsFile = true;
	bool bDestIsFile = true;
	bool bUseGPU = false;
	bool bThreaded = false;
	unsigned int iModelFrames = DEFAULT_MODEL_TRAINING_COUNT;
	double dLearningRate = DEFAULT_LEARNING_RATE;
	detectionAlgorithm algo = ALGO_MOVING;
	
	// Source video information
	unsigned int iInputWidth, iInputHeight, iInputFps;
	
	/** OPTION READ-IN BEGIN **/
	if(cfg.count("help")) {
		cerr << parser.getDescription() << endl;
		return EXIT_SUCCESS;
	}

	if(cfg.count("source")) {
		if(cfg["source"].as<string>() == "CAM")
			bSourceIsFile = false;
		else if(cfg["source"].as<string>() == "DISK")
			bSourceIsFile = true;
		else {
			cerr << ERROR("unrecognized video source") << endl;
			return EXIT_FAILURE;
		}
	}
	
	if(cfg.count("infile"))
		sInfile.assign(cfg["infile"].as<string>());
	
	if(cfg.count("destination")) {
		if(cfg["destination"].as<string>() == "SCREEN") {
			bDestIsFile = false;
			sOutfile = "SCREEN";
		} else if(cfg["destination"].as<string>() == "DISK")
			bDestIsFile = true;
		else {
			cerr << ERROR("unrecognized output destination") << endl;
			return EXIT_FAILURE;
		}
	}
	
	if(bDestIsFile && cfg.count("outfile"))
		sOutfile.assign(cfg["outfile"].as<string>());
	
	if(cfg.count("modelframes")) {
		iModelFrames = cfg["modelframes"].as<unsigned>();
		if(iModelFrames == 0 || iModelFrames > 1000) {
			cerr << ERROR("bad model frame count") << endl;
			return EXIT_FAILURE;
		}
	}
	
	if(cfg.count("learnrate")) {
		dLearningRate = cfg["learnrate"].as<double>();
		if(dLearningRate <= 0.0 || dLearningRate >= 1.0) {
			cerr << ERROR("bad model learning rate") << endl;
			return EXIT_FAILURE;
		}
	}
	
	if(cfg.count("detection")) {
		algo = (detectionAlgorithm)cfg["detection"].as<int>();
		if(algo < ALGO_MOVING || algo > ALGO_CANDIDATESNEW) {
			cerr << parser.getDescription() << endl;
			return EXIT_FAILURE;
		}
	}
	
	if(cfg.count("gpu"))
		bUseGPU = true;
	
	if(cfg.count("threaded"))
		bThreaded = true;
	/** OPTION READ-IN END **/

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
	
	timer start, end;
	double framerate_avg = INITIAL_FPS;
	unsigned long long frames_processed = 1;
	
	// This is the software pipelined implementation of the core tracking loop
	if(bThreaded) {
		Mat frame;
		
		SynchronisedQueue<Mat*> readerToProcessorQ(PIPELINE_BUFFER_SIZE);
		SynchronisedQueue<Mat*> processorToWriterQ(PIPELINE_BUFFER_SIZE);
		
		ReaderThread rt(&cap, &readerToProcessorQ, iInputWidth, iInputHeight);
		ProcessorThread pt(&readerToProcessorQ, &processorToWriterQ, bUseGPU, iInputWidth, iInputHeight, iModelFrames, dLearningRate, algo);
		
		boost::thread reader(rt);
		boost::thread processor(pt);
		
		gettimeofday(&start, NULL);
		
		while(true) {
			Mat* tmp = processorToWriterQ.Dequeue();
			frame = tmp->clone();
			delete tmp;
			vb << frame;
			
			gettimeofday(&end, NULL);
			double throughput = (double)timevaldiff(start, end);
			double maxfps = 1.0/(throughput/1000.0);
			if(maxfps < 1.0)
				maxfps = 1.0;
			framerate_avg *= frames_processed;
			framerate_avg += maxfps;
			frames_processed++;
			framerate_avg /= frames_processed;
			cout << "Instantaneous frame rate: " << maxfps << ", average frame rate: " << framerate_avg << "\r" << flush;
			start = end;
			
			if(!bDestIsFile)
				if(waitKey(1) >= 0) break;
		}
	} else { // This is the single-threaded core tracking loop
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
		
		Mat frame, segmentedFrame, previousImage, dbg;
		bool updateBackground = true;
		bool firstFrame = true;

		while(1) {
			uint32_t candidates[ALLOWED_CANDIDATES+1];
			vector< pair<unsigned,unsigned> > cForegroundList;
			
			// Capture current frame and measure time
			gettimeofday(&start, NULL);
			cap >> frame;
			gettimeofday(&end, NULL);
			double capture = timevaldiff(start,end);
			
			// In case of GPU processing, upload and prepare frame
			gettimeofday(&start, NULL);
			if(bUseGPU) {
				fg.uploadPreprocessFrame(frame);
			}
			gettimeofday(&end, NULL);
			double preprocess = timevaldiff(start,end);
			// Update background model for every second image and measure time
			gettimeofday(&start, NULL);
			if(updateBackground) {
				fg.addFrameToModel(frame);
			}
			updateBackground = !updateBackground;
			gettimeofday(&end, NULL);
			double backgroundadd = timevaldiff(start, end);
			
			// Segment foreground for current frame and measure time
			gettimeofday(&start, NULL);
			fg.segment(frame, segmentedFrame, cForegroundList);
			gettimeofday(&end, NULL);
			double segmentation = timevaldiff(start, end);
			
			// Generate low level ball candidates
			gettimeofday(&start, NULL);
			fg.genLowLevelCandidates(segmentedFrame, candidates);
			gettimeofday(&end, NULL);
			double candgen = bUseGPU ? 0.5 : timevaldiff(start, end); // for some reason, timevaldiff reports 1ms higher time than CUDA events on this stage, so we fix it to 0.5ms as reported by CUDA events
			
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
				bd.searchBall(segmentedFrame, frame, cForegroundList, candidates, algo);
			firstFrame = false;
			gettimeofday(&end, NULL);
			double detection = timevaldiff(start,end);
			
			// Output frame to disk / screen
			gettimeofday(&start, NULL);
			vb << frame;
			gettimeofday(&end, NULL);
			double display = timevaldiff(start, end);
			if(!bDestIsFile)
				if(waitKey(30) >= 0) break;

			// For optical flow based ball detection: update motion reference frame
			previousImage = segmentedFrame.clone();
			bd.updatePreviousImage(&previousImage);
			
// 			cout << "Capture: " << capture << "ms, Preprocess: " << preprocess << "ms, Background Update: " << backgroundadd << "ms, Segmentation: "<< segmentation << "ms, CandidateGeneration: " << candgen << "ms,  Detection: "<< detection << "ms, Display: " << display << "ms" << endl;
			
			// Calculate maximum sustainable frame rate
			double total = preprocess + backgroundadd + segmentation + detection + candgen;// + capture + display;
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
		}
	}
	
	return 0;
}


