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

//our grid
#include "GridC.h"
#include "GridP.h"
#include "Candidate.h"
#include "Path.h"

#define HTHRESHOLD 30
#define LTHRESHOLD 16

#define BALL_PELS_UPPER 50
#define BALL_PELS_LOWER 10

#define MINCHAINLENGTH 7
#define MINSTARTENDDISTANCE 25
#define MAXSTEPDISTANCE IGNORERADIUS

#define MAXPATHSWITHSAMESTART 5

using namespace cv;
using namespace std;

/*! Detection algorithm decision parameter */
enum detectionAlgorithm {
	ALGO_MOVING,
	ALGO_CLUSTER,
	ALGO_OPTICAL,
	ALGO_CIRCLES,
	ALGO_CANDIDATES,
	ALGO_CANDIDATESNEW,
};

/*!	Class: BallDetection
 *
 *	Contains ball detection algorithms to (generate and) evaluate possible ball candidates,
 * 	i.e. discriminate the ball from the players. Ball detection is realized entirely on the
 * 	CPU. The various algorithms differ in their computational complexity, sophistication
 * 	as well as detection accuracy.
 */
class BallDetection{
public:
	/*! CTOR */
	BallDetection();
	
	/*!	Function: setImageParams.
	 * 
	 *	Specify the parameters of the images being processed.
	 */
	void setImageParams(unsigned int _width, unsigned int _height, unsigned int _channels);

	/*!	Function: toggleBackground.
	 * 
	 *	The background of the output image can be selected by calling this function.
	 */
	void toggleBackground(void);
	
	/*! Function: searchBall.
	 * 
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
	 * 5) Grid-based with path tracing
	 * 
	 * 6) Grid-based with kalman filter
	 * 
	 * Currently, all algorithms are executed on the CPU.
	 */
	void searchBall(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList, uint32_t * candidates, detectionAlgorithm algo);
	
	/*! Function: updatePreviousImage.
	 * 
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

	//Selected background
	int background;
	
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
	
	//pointer to our grids (new grid solution)
	GridC * candidategrid;
	GridP * pathgrid;
	
	pair<unsigned, unsigned> guessCurrentPositionBasedOnOldPositions();
	pair<unsigned, unsigned> getLastKnownPossition();
	pair<unsigned, unsigned> mostPlausibleCandidate(Mat allCandidates);
	void addCoordinatesToVector(int recognizedX, int recognizedY);
	
	/*! Function: searchpixelatdistancebycolor.
	 * 
	 * This function is used by the moving mask algorithm.
	 * It searches a pixel with specified intensity of a specified channel in the neighbourhood of specified starting position.
	 * Its runs around the starting position in circles until if finds a matching pixel or the specified distance is exceeded.
	 * The function returns 1 if a matching pixel was found or 0 otherwise.
	 * arguments:
	 * 	image: an image
	 * 	channel: channel (0 for blue, 1 for green, 2 for red)
	 *	intensity: the intensity of the pixel at a specified channel to look for
	 * 	x, y: coordinates of the starting position
	 *	offsetx, offsety: pointers to store the offset from the starting position to the first matching pixel
	 *	dist: maximal distance
	 */	
	int searchpixelatdistancebycolor(Mat * image, int channel, int color, int x, int y, int * offsetx, int * offsety, int dist);

	/*! Function: searchpixelatdistancebycolormeanresults.
	 * 
	 * This function is used by the circle based algorithm.
	 * It searches pixels with specified intensity of a specified channel in the neighbourhood of specified starting position.
	 * Its runs around the starting position in circles until if finds a distance, where one or more matching pixels are present,
	 * or the specified distance is exceeded.
	 * The function returns the amount of detected pixels with the first distance with matching pixels
	 * or 0 if the distance limit is reached but no matching pixels were found.
	 * arguments:
	 * 	image: an image
	 * 	channel: channel (0 for blue, 1 for green, 2 for red)
	 *	intensity: the intensity of the pixel at a specified channel to look for
	 * 	x, y: coordinates of the starting position
	 *	offsetx, offsety: pointers to store the mean offset from the starting position to the matching pixels
	 *	dist: maximal distance
	 */
	int searchpixelatdistancebycolormeanresults(Mat * image, int channel, int color, int x, int y, int * offsetx, int * offsety, int dist);

	/*! Function: countblackpixelsaroundatedge.
	 * 
	 * This function is used by the circle based algorithm.
	 * It counts dark pixels at the edge of a square of a specified size and a specified center.
	 * It returns the amount of the detected dark pixels.
	 * arguments:
	 * 	image: an image
	 * 	centerx, centery: coordinates of the center of the square
	 *	halfsquaresize: half of the length of the side of the square in pixels
	 */
	int countblackpixelsaroundatedge(Mat * image, int centerx, int centery, int halfsquaresize);
	
	/*! Function: locateBallMovingMask.
	 * 
	 * This function searches for the ball using the algorithm with the moving mask.
	 * arguments:
	 * 	srcFrame: the image with foreground pixels
	 * 	dstFrame: the output image
	 *	cForegroundList: list of the foreground pixels
	 */
	void locateBallMovingMask(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList);

	/*! Function: locateBallOpticalFlow.
	 * 
	 * This function tries to detect the ball by performing motion estimation for all the foreground pixels.
	 * Given the motion for each pixel, the absolute pixel displacement between the current and the reference
	 * frame is computed. If it is above a threshold, the pixel is assumed to belong to the ball.
	 * 
	 * Arguments:
	 * 	srcFrame: the image with foreground pixels
	 * 	dstFrame: the output image
	 *	cForegroundList: list of the foreground pixels
	 */
	void locateBallOpticalFlow(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList);
	
	/*!	Function: locateBallForegroundClusters.
	 * 
	*	detectes ball by clustering all foreground pixel and afterswards using thresholds to 
	* 	decide between ball and players. It uses the (5) past clustercenter for deciding.
	* @srcFrame = original Image
	* @dstFrame = result Image
	* @cForegroundList = list of Foreground pixel
	*/
	void locateBallForegroundClusters(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList);

	/*! Function: locateBallHistoryBased.
	 * 
	 * This function searches for the ball using the history based algorithm.
	 * arguments:
	 * 	srcFrame: the image with foreground pixels
	 * 	dstFrame: the output image
	 *	cForegroundList: list of the foreground pixels
	 */
	void locateBallHistoryBased(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned> >& cForegroundList);

	/*! Function: locateBallCandidatesOld.
	 * 
	 * This function searches for the ball using the circle based algorithm with a list of candidates instead of the foreground pixels.
	 * arguments:
	 * 	srcFrame: the image with foreground pixels
	 * 	dstFrame: the output image
	 *	candidates: array containing the list of candidates
	 */
	void locateBallCandidatesOld(Mat& srcFrame, Mat& dstFrame, uint32_t * candidates);

	/*! Function: locateBallCandidatesNew.
	 * 
	 * This function searches for the ball using the grid based algorithm with a list of candidates.
	 * arguments:
	 * 	srcFrame: the image with foreground pixels
	 * 	dstFrame: the output image
	 *	candidates: array containing the list of candidates
	 */
	void locateBallCandidatesNew(Mat& srcFrame, Mat& dstFrame, uint32_t * candidates);
	
	/*! Function: putcandidatestogrid.
	 * 
	 * This function is used by the grid based algorithm. It puts new candidates to the according grid cells.
	 * arguments:
	 * 	candidates: array containing the list of candidates
	 */
	void putcandidatestogrid(uint32_t * candidates);

	/*! Function: removeorrelocatepaths.
	 * 
	 * This function is used by the grid based algorithm.
	 * It removes all old paths which have not have been connected with a new candidate
	 * and paths out of range of the picture from the according grid cells.
	 * It also checks weither the path came out our range of their cells after they have been connected
	 * with new cadidates and relocates the paths to other grid cells if required.
	 */
	void removeorrelocatepaths(void);

	/*! Function: clearcandidategrid.
	 * 
	 * This function is used by the grid based algorithm.
	 * It removes all old candidates from the candidate grid.
	 */
	void clearcandidategrid(void);

	/*! Function: insertclosestcandidate.
	 * 
	 * This function is used by the grid based algorithm.
	 * It tries to insert the closest new candidate to a path. It checks all gridcells in the specified maxmal range to do so.
	 * arguments:
	 *	p: pointer to the path
	 *	minradius: minimal distance between the current beginning of the path and the new candidate
	 *	maxradius: maximal distance between the current beginning of the path and the new candidate
	 */
	void insertclosestcandidate(Path * p, int minradius, int maxradius);

	/*! Function: insertclosestcandidateinallpaths.
	 * 
	 * This function is used by the grid based algorithm.
	 * It goes through all paths in the whole path grid and tries to connect them with the closest candidates.
	 * arguments:
	 *	minradius: minimal distance between the current beginning of the path and the new candidate
	 *	maxradius: maximal distance between the current beginning of the path and the new candidate
	 */
	void insertclosestcandidateinallpaths(int minradius, int maxradius);

	/*! Function: insertnewpath.
	 * 
	 * This function is used by the grid based algorithm.
	 * It creates a new path with the specified candidate at the beginning and inserts it to the according cell of the path grid.
	 * arguments:
	 *	start: the new candidate at the start of the new path
	 */
	void insertnewpath(Candidate start);

	/*! Function: createnewpathsfromremainingcandidates.
	 * 
	 * This function is used by the grid based algorithm.
	 * It creates a new paths from all new candidates except the marked candidates.
	 */
	void createnewpathsfromremainingcandidates(void);

	/*! Function: removepathswithsamestart.
	 * 
	 * This function is used by the grid based algorithm.
	 * It goes through all candidates in the candidate grid and checks the amount of paths they were connected to.
	 * If a candidate has to many paths connected to it then all thiese paths are removed from the pathgrid.
	 */
	void removepathswithsamestart(void);

	/*! Function: vectlength.
	 * 
	 * This function returns the squared length of a two dimensional vector with specified components.
	 * arguments:
	 *	x, y: components of the vector
	 */
	double vectlength(double x, double y) {
		return x*x+y*y;
	}

	/*! Function: diffabs.
	 * 
	 * This function returns the squared length of a two dimensional vector with specified components.
	 * arguments:
	 *	x, y: components of the vector
	 */	
	float diffabs(float x1, float y1, float x2, float y2) {
		float diffx = x1 - x2;
		float diffy = y1 - y2;
		return diffx*diffx+diffy*diffy;
	}
};

#endif
