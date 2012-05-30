#ifndef BALL_DETECTION_H
#define BALL_DETECTION_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>

#include <vector>
#include <list>
#include <utility>
#include <iostream>
#include <algorithm>

// #define HALFTABLESIZE 20
// #define MINRADIUS 3
// #define MAXRADIUS 5
// #define IGNORERADIUS 8
#define HTHRESHOLD 30
#define LTHRESHOLD 16

#define BALL_PELS_UPPER 50
#define BALL_PELS_LOWER 10

using namespace cv;
using namespace std;

enum detectionAlgorithm {
	ALGO_MOVING,
	ALGO_CLUSTER,
	ALGO_OPTICAL
};

class BallDetection{
public:
	BallDetection();
	void setImageParams(unsigned int _width, unsigned int _height, unsigned int _channels);
	void searchBall(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList, detectionAlgorithm algo);
  
	void updatePreviousImage(Mat* frame) {
		lastImage = frame;
	}
	
	pair<unsigned, unsigned> guessCurrentPositionBasedOnOldPositions();
	pair<unsigned, unsigned> getLastKnownPossition();
	pair<unsigned, unsigned> mostPlausibleCandidate(Mat allCandidates);
	void addCoordinatesToVector(int recognizedX, int recognizedY);
    
private:
	unsigned int iBgWidth, iBgHeight, iBgChannels;
	unsigned int countframes;
// 	int table[HALFTABLESIZE*2+1][HALFTABLESIZE*2+1];
	Mat* lastImage;
	vector<Mat> pastCenterCoordinates;
	list<Point2f> vPreviousBallLocations;
	bool tracking;
	unsigned long trackedFrames;
	int iNumberOfClusters; //expect cluster for 2 player + 1 ball times 2 for a possible 2nd pair
	int amountOfPastCenterCoords;
	
	Mat path;
	int HALFTABLESIZE;
	int MINRADIUS;
	int MAXRADIUS;
	int IGNORERADIUS;
	
	int lookForConsistantTextures(int startRow, int startCol,  Mat& srcFrame);	
	void locateBallMovingMask(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList);
	void locateBallOpticalFlow(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList);
	void locateBallForegroundClusters(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList);
};

#endif
