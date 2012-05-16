#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>

#include <vector>
#include <utility>
#include <iostream>

#define HALFTABLESIZE 20
#define MINRADIUS 3
#define MAXRADIUS 5
#define IGNORERADIUS 8
#define HTHRESHOLD 128
#define LTHRESHOLD 64

#define BALL_PELS_UPPER 60
#define BALL_PELS_LOWER 20

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
	void searchBall(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned>>& cForegroundList, detectionAlgorithm algo);
  
	void updatePreviousImage(Mat* frame) {
		lastImage = frame;
	}
    
private:
	unsigned int iBgWidth, iBgHeight, iBgChannels;
	unsigned int countframes;
	int table[HALFTABLESIZE*2+1][HALFTABLESIZE*2+1];
	Mat* lastImage;
	vector<Mat> pastCenterCoordinates;
	int iNumberOfClusters; //expect cluster for 2 player + 1 ball times 2 for a possible 2nd pair
	int amountOfPastCenterCoords;
	
	int lookForConsistantTextures(int startRow, int startCol,  Mat& srcFrame);	
	void locateBallMovingMask(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned>>& cForegroundList);
	void locateBallOpticalFlow(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned>>& cForegroundList);
	void locateBallForegroundClusters(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned>>& cForegroundList);
};
