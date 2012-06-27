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

#define MINCHAINLENGTH 7
#define MINSTARTENDDISTANCE 25
#define MAXSTEPDISTANCE IGNORERADIUS

using namespace cv;
using namespace std;

// Detection algorithm decision parameter
enum detectionAlgorithm {
	ALGO_MOVING,
	ALGO_CLUSTER,
	ALGO_OPTICAL,
	ALGO_CIRCLES,
	ALGO_CANDIDATES,
	ALGO_CANDIDATESNEW,
};

// List element for grid entries
struct listelement {
	struct listelement * next;
	struct listelement * closestfromnextlayer;
	int x, y, t;
	int chainendx, chainendy, chainlength;
};

// Class containing ball detection algorithms
class BallDetection{
public:
	// CTOR
	BallDetection();
	
	/*	Function: setImageParams
	 *	------------------------
	 *	Specify the parameters of the images being processed.
	 */
	void setImageParams(unsigned int _width, unsigned int _height, unsigned int _channels);
	
	/* Function: searchBall
	 * --------------------
	 * Central function of the ball detection procedure. Takes a source 
	 * frame (segmented or otherwise informative frame) and a destination
	 * frame (frame to draw the detected balls into, i.e. original frame)
	 * along with a list of foreground points.
	 * 
	 * The detection algorithm decides which algorithm will be used:
	 * 
	 * 1) Motion estimation / Optical flow
	 * 
	 * 2) K-Means clustering
	 * 
	 * 3) Template fitting / "Generalized Hough Transform"
	 * 
	 * 4) Path tracing based
	 * 
	 * Currently, all algorithms are executed on the CPU. The best performance
	 * in FPS*ACCURACY is delivered by the path tracer.
	 */
	void searchBall(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList, uint32_t * candidates, detectionAlgorithm algo);
	
	/* Function: updatePreviousImage
	 * -----------------------------
	 * Used for optical flow based ball discrimination. Sets
	 * the reference image for motion estimation.
	 */
	void updatePreviousImage(Mat* frame) {
		lastImage = frame;
	}    
private:
	// General information
	unsigned int iBgWidth, iBgHeight, iBgChannels;
	unsigned int countframes;
	
	// Optical Flow based detection
	Mat* lastImage;
	bool tracking;
	unsigned long trackedFrames;
	
	// Cluster-based detection
	vector<Mat> pastCenterCoordinates;
	list<Point2f> vPreviousBallLocations;
	int iNumberOfClusters; //expect cluster for 2 player + 1 ball times 2 for a possible 2nd pair
	int amountOfPastCenterCoords;
	int iCounterOfNoFoundFrames;
	vector<pair<unsigned, unsigned>> pastCoordinatesOfRecognizedBalls;
	
	// For path-based and circle-based detection
	Mat path;
	int HALFTABLESIZE;
	int MINRADIUS;
	int MAXRADIUS;
	int IGNORERADIUS;
	double oldBallPositions[8][2];
	int currentInsertPosition;
	
	// For grid-based detection
	int framecnt;
	struct listelement * grid[53][30][8]; //Grid of (848/16=53)x(480/16=30) cells and 8 time layers.
	
	pair<unsigned, unsigned> guessCurrentPositionBasedOnOldPositions();
	pair<unsigned, unsigned> getLastKnownPossition();
	pair<unsigned, unsigned> mostPlausibleCandidate(Mat allCandidates);
	void addCoordinatesToVector(int recognizedX, int recognizedY);
	int lookForConsistantTextures(int startRow, int startCol,  Mat& srcFrame);
	
	int searchpixelatdistancebycolor(Mat * image, int channel, int color, int x, int y, int * offsetx, int * offsety, int dist);
	int searchpixelatdistancebycolormeanresults(Mat * image, int channel, int color, int x, int y, int * offsetx, int * offsety, int dist);
	int countblackpixelsaroundatedge(Mat * image, int centerx, int centery, int halfsquaresize);
	
	void grid_init(void);
	void cleargridcell(int x, int y, int t);
	void cleargridlayer(int t);
	void putcandidatestogrid(uint32_t * candidates, int t);
	void connecttoclosestfromprevlayer(listelement * thiselement);
	void connectallatcurrentlayer(int t);
	listelement * longestchainstart(int t);
	
	void locateBallMovingMask(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList);
	void locateBallOpticalFlow(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList);
	void locateBallForegroundClusters(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList);
	void locateBallCircleBased(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList);
	void locateBallCandidatesOld(Mat& srcFrame, Mat& dstFrame, uint32_t * candidates);
	void locateBallCandidatesNew(Mat& srcFrame, Mat& dstFrame, uint32_t * candidates);
	
	double vectlength(double x, double y) {
		return x*x+y*y;
	}
	
	float diffabs(float x1, float y1, float x2, float y2) {
		float diffx = x1 - x2;
		float diffy = y1 - y2;
		return diffx*diffx+diffy*diffy;
	}
};

#endif
