#include "BallDetection.h"
#include "TrackerExceptions.h"

#include "global.h"

bool cmpPointPairX(Point2f a, Point2f b) {
	return (a.x < b.x);
}

bool cmpPointPairY(Point2f a, Point2f b) {
	return (a.y < b.y);
}

BallDetection::BallDetection() {
	lastImage = NULL;
	countframes = 0;

	iNumberOfClusters = 6;
	amountOfPastCenterCoords = 5;
	iCounterOfNoFoundFrames = 0;
	
	tracking = false;
	
	currentInsertPosition = 0;

	HALFTABLESIZE = 7; 
	MINRADIUS = 3;
	MAXRADIUS = 5;
	IGNORERADIUS = 15;

	background = 0;
};

void BallDetection::setImageParams(unsigned int _width, unsigned int _height, unsigned int _channels) {
	iBgWidth = _width;
	iBgHeight = _height;
	assert(_channels == 1);
	iBgChannels = _channels;

	//Create a picture to draw the history
	path = Mat(iBgHeight, iBgWidth, CV_8UC3, Scalar(0.0)).clone();

	//Create the grids and initialise all their entries with empty lists
	candidategrid = new GridC(iBgWidth, iBgHeight);
	pathgrid = new GridP(iBgWidth, iBgHeight);
	for(int x=0; x<pathgrid->xcells; x++) {
		for(int y=0; y<pathgrid->ycells; y++) {
			candidategrid->SetEntry(x, y, new list<Candidate>());
			pathgrid->SetEntry(x, y, new list<Path>());
		}
	}
}

void BallDetection::toggleBackground(void) {
	//Switch the background by incrementing the variable and setting it back to zero when it reaches 3
	background++;
	background %= 3;
}

void BallDetection::searchBall(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned>>& cForegroundList, uint32_t * candidates, detectionAlgorithm algo){
	//Call an according function depending on the selected algorithm
	switch(algo) {
		case ALGO_MOVING:
			locateBallMovingMask(srcFrame, dstFrame, cForegroundList);
			break;
		case ALGO_CLUSTER:
			locateBallForegroundClusters(srcFrame, dstFrame, cForegroundList);
			break;
		case ALGO_OPTICAL:
			locateBallOpticalFlow(srcFrame, dstFrame, cForegroundList);
			break;
		case ALGO_CIRCLES:
			locateBallHistoryBased(dstFrame, dstFrame, cForegroundList);
			break;
		case ALGO_CANDIDATES:
			locateBallCandidatesOld(srcFrame, dstFrame, candidates);
			break;
		case ALGO_CANDIDATESNEW:
			locateBallCandidatesNew(srcFrame, dstFrame, candidates);
			break;
		default:
			throw IllegalArgumentException();
	}
}

void BallDetection::locateBallOpticalFlow(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned>>& cForegroundList) {
	if(cForegroundList.size() == 0)
		return;
	
	vector<uchar> status;
	vector<float> err;
	vector<float> distances;
	vector<Point2f> currentPoints;
	vector<Point2f> currentFilteredPoints;
	vector<Point2f> nextPoints;
	vector<Point2f> nextFilteredPoints;
	unsigned count = 0;
	
	for(vector<pair<unsigned,unsigned>>::iterator it = cForegroundList.begin(); it != cForegroundList.end(); ++it) {
		currentPoints.push_back(Point2f((*it).first, (*it).second));
	}
	srcFrame.convertTo(srcFrame, CV_8UC1);
	lastImage->convertTo(*lastImage, CV_8UC1);
	
	cv::calcOpticalFlowPyrLK(*lastImage, srcFrame, currentPoints, nextPoints, status, err);
	
	vector<Point2f>::iterator itnext = nextPoints.begin();
	for(vector<Point2f>::iterator it = currentPoints.begin(); it != currentPoints.end(); ++it) {
// 		cout << "(" << it->x << "," << it->y << ") -> (" << itnext->x << "," << itnext->y << ")" << endl;
		float traveledDistance = (it->x-itnext->x)*(it->x-itnext->x) + (it->y-itnext->y)*(it->y-itnext->y);
		distances.push_back(traveledDistance);
// 		cout << traveledDistance << endl;
		if(traveledDistance > 0.4f) {
			count++;
			currentFilteredPoints.push_back(Point2f(it->x, it->y));
			nextFilteredPoints.push_back(Point2f(itnext->x, itnext->y));
		}
		itnext++;
	}

// 	bool detected = (count > BALL_PELS_LOWER && count < BALL_PELS_UPPER);

	// If ball was detected with reasonable accuracy, draw it and add current position to past ball positions
// 	if(detected) {
// 		cout << "Ball detected at " << count << " Pixels." << endl;
		// Draw ball motion
		itnext = nextFilteredPoints.begin();
		bool drawLine = true;
		for(vector<Point2f>::iterator it = currentFilteredPoints.begin(); it != currentFilteredPoints.end(); it++) {
			if(drawLine)
				line(dstFrame, *it, *itnext, Scalar(0,0,255), 1, 8);
			drawLine = !drawLine;
			itnext++;
		}
		
		// Remove previous estimated results
// 		if(!tracking) {
// 			vPreviousBallLocations.clear();
// 			tracking = true;
// 		}
			
		// Find ball corner, find minimum in both x and y direction
// 		float x, y;
// 		sort(currentFilteredPoints.begin(), currentFilteredPoints.end(), cmpPointPairX);
// 		x = currentFilteredPoints[9].x;
// 		sort(currentFilteredPoints.begin(), currentFilteredPoints.end(), cmpPointPairY);
// 		y = currentFilteredPoints[9].y;
// 		cout << "Located at " << x << "," << y << endl;
// 		vPreviousBallLocations.push_front(Point2f(x,y));
// 	} else {
		// If no good ball candidate, draw prediction from past candidates and delete past candidates
// 		tracking = false;
// 		
// 		// Linear interpolation
// 		if(vPreviousBallLocations.size() >= 2) {
// 			list<Point2f>::iterator it = vPreviousBallLocations.begin();
// 			Point2f last = *it++;
// 			Point2f secondToLast = *it;
// 			Point2f nextPoint;
// 			nextPoint.x = secondToLast.x + (last.x - secondToLast.x)*(2);
// 			nextPoint.y = secondToLast.y + (last.y - secondToLast.y)*(2);
// 			vPreviousBallLocations.push_front(nextPoint);
// // 			cout << "Next point " << nextPoint.x << "," << nextPoint.y << " from " << last.x << "," << last.y << " and " << secondToLast.x << "," << secondToLast.y << endl;
// 		}
// 	}
	
	// Draw flight contours
// 	if(vPreviousBallLocations.size() >= 2) {
// 		list<Point2f>::iterator nextit, tmp;
// 		for(list<Point2f>::iterator it = vPreviousBallLocations.begin(); it != vPreviousBallLocations.end(); it++) {
// 			tmp = it;
// 			nextit = it++;
// 			it = tmp;
// 			if(nextit == vPreviousBallLocations.end())
// 				break;
// 			line(dstFrame, *it, *nextit, Scalar(0,0,255), 1, 8);				
// 		}
// 	}
	return;
}

int iCounterOfNoFoundFrames = 0;
void BallDetection::locateBallForegroundClusters(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned>>& cForegroundList) {
// 	cout << "Clustering" <<endl;
	int iCounter = 0;
	Mat xyMatrix = Mat::zeros(iBgHeight*iBgWidth, 2, CV_32F);
	int iWindowSize = 50*iCounterOfNoFoundFrames+10;
	pair<unsigned, unsigned> temp = getLastKnownPossition();
	int iLastKnownPositionX = temp.first;
	int iLastKnownPositionY = temp.second;
	
	int iBottumLimitX, iTopLimitX, iBottumLimitY, iTopLimitY;
	if(iLastKnownPositionX != 0 && iLastKnownPositionY != 0){
	  iBottumLimitX = max(0,iLastKnownPositionX - iWindowSize);
	  iBottumLimitY = max(0, iLastKnownPositionY -iWindowSize);
	  iTopLimitX = iLastKnownPositionX + iWindowSize;
	  iTopLimitY = iLastKnownPositionY + iWindowSize;
	}else{
	  iBottumLimitX = 0;
	  iBottumLimitY = 0;
	  iTopLimitX = iBgWidth;
	  iTopLimitY = iBgHeight;
	}

	
	for(vector< pair<unsigned,unsigned> >::const_iterator it = cForegroundList.begin(); it != cForegroundList.end(); ++it) {
		if((*it).first > iBottumLimitX && (*it).first < iTopLimitX && (*it).second > iBottumLimitY && (*it).second < iTopLimitY){
		  //add it only, if it is not to far from last known position
		  xyMatrix.at<float>(iCounter, 1) = (*it).first;
		  xyMatrix.at<float>(iCounter, 0) = (*it).second;
		  iCounter++;
		}
	}
	int iMatrixSize = iCounter;
	
	double dEpsilon = 3.0; 
	int iAmountOfIterations = 10;		//amount of iterations of kmeans
	int iAmountOfAttemps = 1; 		//amount of new cluster init points
  
	TermCriteria termCriteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, iAmountOfIterations, dEpsilon);

	//init result matrixes
	Mat mCoordinatesOfCenters = Mat::zeros(iNumberOfClusters, 2, CV_32FC1);
	
	Mat mLabels(xyMatrix.rows, 1, CV_32FC1);

	//calculate kmeans and giving labeled result matrixes as well as center of clusters
	kmeans(xyMatrix, iNumberOfClusters, mLabels, termCriteria, iAmountOfAttemps, KMEANS_PP_CENTERS, mCoordinatesOfCenters);
	
	//keep history small
	if(pastCenterCoordinates.size() > 5){
		pastCenterCoordinates.erase(pastCenterCoordinates.begin(), pastCenterCoordinates.begin());
	}
     
	iCounter = 0;
  
	int sizeOfCluster[iNumberOfClusters];
	for(int i = 0; i < iNumberOfClusters; i++){
		sizeOfCluster[i] = 0;
	}
	int lowerLimits[2][iNumberOfClusters];
	int upperLimits[2][iNumberOfClusters];
	for(int i = 0; i < iNumberOfClusters; i++){
	  lowerLimits[0][i] = 2000;
	  lowerLimits[1][i] = 2000;
	  upperLimits[0][i] = 0;
	  upperLimits[1][i] = 0;
	}
	
	int differencesX[iNumberOfClusters];
	int differencesY[iNumberOfClusters];
  
	for(int i = 0; i< iMatrixSize; i++){
	  int iTempLabelNumber = mLabels.at<uchar>(i, 0);
	  int iColor [3];
	  switch(iTempLabelNumber){
		  case 1:
			  
			  sizeOfCluster[iTempLabelNumber-1]++;
			  iColor[0] = 255;
			  iColor[1] = 0;
			  iColor[2] = 0;
			  break;
		  case 2:
			  sizeOfCluster[iTempLabelNumber-1]++;
			  iColor[0] = 0;
			  iColor[1] = 255;
			  iColor[2] = 0;
			  break;
		  case 3:
			  sizeOfCluster[iTempLabelNumber-1]++;
			  iColor[0] = 0;
			  iColor[1] = 0;
			  iColor[2] = 255;
			  break;
		  case 4:
			  sizeOfCluster[iTempLabelNumber-1]++;
			  iColor[0] = 255;
			  iColor[1] = 255;
			  iColor[2] = 0;
			  break;
		  case 5:
			  sizeOfCluster[iTempLabelNumber-1]++;
			  iColor[0] = 255;
			  iColor[1] = 0;
			  iColor[2] = 255;
			  break;
		  case 6:
			  sizeOfCluster[iTempLabelNumber-1]++;
			  iColor[0] = 0;
			  iColor[1] = 255;
			  iColor[2] = 255;
			  break;
		  default:
			  iColor[0] = 255;
			  iColor[1] = 255;
			  iColor[2] = 255;
			  break;
	  }
	  if(xyMatrix.at<float>(i,0) < lowerLimits[0][iTempLabelNumber-1]){
	    lowerLimits[0][iTempLabelNumber-1] = xyMatrix.at<float>(i,0);
	  }
	  if(xyMatrix.at<float>(i,0) > upperLimits[0][iTempLabelNumber-1]){
	    upperLimits[0][iTempLabelNumber-1]= xyMatrix.at<float>(i,0);
	  }
	  if(xyMatrix.at<float>(i,1) < lowerLimits[1][iTempLabelNumber-1]){
	  lowerLimits[1][iTempLabelNumber-1]= xyMatrix.at<float>(i,1);
	  }
	  if(xyMatrix.at<float>(i,1) > upperLimits[1][iTempLabelNumber-1]){
	    upperLimits[1][iTempLabelNumber-1]= xyMatrix.at<float>(i,1);
	  }
	    
	  int iTempX = xyMatrix.at<float>(i,1);
	  int iTempY = xyMatrix.at<float>(i,0);
	}
	pair<unsigned, unsigned> CenterPos = mostPlausibleCandidate(mCoordinatesOfCenters);
	Point center(CenterPos.first,CenterPos.second);	
	circle(dstFrame, center, 5, Scalar(0,0,255), 2,4,0);
	
	addCoordinatesToVector(CenterPos.first, CenterPos.second);
	pastCenterCoordinates.push_back(mCoordinatesOfCenters);
}

void BallDetection::addCoordinatesToVector(int recognizedX, int recognizedY){
    if(pastCoordinatesOfRecognizedBalls.size() > 15){
     pastCoordinatesOfRecognizedBalls.erase(pastCoordinatesOfRecognizedBalls.begin()); 
    }
    //insert at last position
    pastCoordinatesOfRecognizedBalls.push_back(make_pair(recognizedX, recognizedY));
}

pair<unsigned, unsigned> BallDetection::mostPlausibleCandidate(Mat allCandidates){
      //something like k-means
      double distance[allCandidates.rows];
      int smallestCandidate = 0;
      if(pastCoordinatesOfRecognizedBalls.size() >= 1){
	  for(int i = 0; i < allCandidates.rows; i++){
	      distance[i] = 0;
	      int iCounter = 0;
	      for(vector< pair<unsigned,unsigned> >::const_iterator it = pastCoordinatesOfRecognizedBalls.begin(); it != pastCoordinatesOfRecognizedBalls.end(); ++it){
		  int deltaX = (*it).first - allCandidates.at<float>(i,1);
		  int deltaY = (*it).second - allCandidates.at<float>(i,0);
		  distance[i] += sqrt(deltaX*deltaX + deltaY*deltaY);
		  iCounter++;
	      }	
	      distance[i] /= (double)iCounter;
	      if(i!=0){
		  if(distance[i-1] > distance[i]){
		      smallestCandidate = i;
		  }
	      }
	  }
	  if(distance[smallestCandidate] == 0){
	      return guessCurrentPositionBasedOnOldPositions();
	  }
	  return pair<unsigned, unsigned>(allCandidates.at<float>(smallestCandidate,1), allCandidates.at<float>(smallestCandidate,0));
      }
      return pair<unsigned, unsigned>(allCandidates.at<float>(1,1),allCandidates.at<float>(1,0));
}
pair<unsigned, unsigned> BallDetection::getLastKnownPossition(){
    if(pastCoordinatesOfRecognizedBalls.size()>15){
	return pastCoordinatesOfRecognizedBalls.at(pastCoordinatesOfRecognizedBalls.size()-1);	
    }else{
	return pair<unsigned, unsigned>(0,0); 
    }
}

pair<unsigned, unsigned> BallDetection::guessCurrentPositionBasedOnOldPositions(){
    //take last 3 recognized Values and make quadratic extrapolation
    if(pastCoordinatesOfRecognizedBalls.size() >=15){
	pair<unsigned, unsigned> firstCoord = pastCoordinatesOfRecognizedBalls.at(pastCoordinatesOfRecognizedBalls.size()-2);
	pair<unsigned, unsigned> secondCoord = pastCoordinatesOfRecognizedBalls.at(pastCoordinatesOfRecognizedBalls.size()-1);
	//cout << "Guessing new position" << endl;
    
    int x1 = firstCoord.first;
    int y1 = firstCoord.second;
    int x2 = secondCoord.first;;
    int y2 = secondCoord.second;;
    
    /*
     * position 1: (a,x)
     * position 2: (b,y)
     * 
     * (b-a), (y-x)		newY = (y-x)/(b-a)*newX + lastPoint
     * newX = b+(b-a)
     * 
    */
      int nenner = y2-y1;
      int iGuessForNewX = x2+(x2-x1);
      int iGuessForNewY = y2;
      if(nenner!=0){
	 iGuessForNewY = (x2-x1)/(double) nenner *iGuessForNewX;// + y2;
      }
      if(iGuessForNewX > iBgWidth || iGuessForNewX < 0){
	  iGuessForNewX = 0;
      }
      if(iGuessForNewY > iBgHeight || iGuessForNewY < 0){
	  iGuessForNewY = 0;
      }
	addCoordinatesToVector(iGuessForNewX, iGuessForNewY);
      return pair<unsigned, unsigned>(iGuessForNewX, iGuessForNewY);
    }
    return pair<unsigned, unsigned>(250,250);
}

int BallDetection::searchpixelatdistancebycolor(Mat * image, int channel, int color, int x, int y, int * offsetx, int * offsety, int dist) {
	//Start looking for the pixel at the offset
	int xl = *offsetx;
	int yl = *offsety;

	int startx = *offsetx;
	int starty = *offsety;

	//If the offset and the center are equal then start at the top left corner
	if((startx == x) && (starty == y)) {
		xl = x - dist;
		yl = y - dist;
	} else {
		xl = startx;
		yl = starty;
	}

	//If at the top left corner then search the pixel by moving right until you reach top right corner
	if((xl < x) && (yl < y)) {
		for(;xl<=(x+dist);xl++) {
			if(!(yl >= 0 && yl < iBgHeight && xl < iBgWidth && xl >= 0)) break;
			if((*image).at<Vec3b>(yl, xl)[channel] == color) {
				if((startx != xl) || (starty != yl)) {
					*offsetx = xl;
					*offsety = yl;
					return 1;
				}
			}
		}
		xl = x + dist;
	}

	//If at the top right corner then search the pixel by moving down until you reach bottom right corner
	if((xl > x) && (yl < y)) {		
		for(;yl<=(y+dist);yl++) {
			if(!(yl >= 0 && yl < iBgHeight && xl < iBgWidth && xl >= 0)) break;
			if((*image).at<Vec3b>(yl, xl)[channel] == color) {
				if((startx != xl) || (starty != yl)) {
					*offsetx = xl;
					*offsety = yl;
					return 1;
				}
			}
		}
		yl = y + dist;
	}

	//If at the bottom right corner then search the pixel by moving left until you reach botom left corner
	if((xl > x) && (yl > y)) {
		for(;xl>=(x-dist);xl--) {
			if(!(yl >= 0 && yl < iBgHeight && xl < iBgWidth && xl >= 0)) break;
			if((*image).at<Vec3b>(yl, xl)[channel] == color) {
				if((startx != xl) || (starty != yl)) {
					*offsetx = xl;
					*offsety = yl;
					return 1;
				}
			}
		}
		xl = x - dist;
	}

	//If at the botom left corner then search the pixel by moving up until you reach top left corner
	if((xl < x) && (yl > y)) {
		for(;yl>=(y-dist);yl--) {
			if(!(yl >= 0 && yl < iBgHeight && xl < iBgWidth && xl >= 0)) break;
			if((*image).at<Vec3b>(yl, xl)[channel] == color) {
				if((startx != xl) || (starty != yl)) {
					*offsetx = xl;
					*offsety = yl;
					return 1;
				}
			}
		}
		yl = y - dist;
	}

	//Nothing was found
	return 0;
}

int BallDetection::searchpixelatdistancebycolormeanresults(Mat * image, int channel, int color, int x, int y, int * offsetx, int * offsety, int dist) {
	int xl;
	int yl;
	int rescounter = 0;

	//Start at the top left corner
	xl = x - dist;
	yl = y - dist;

	*offsetx = 0;
	*offsety = 0;

	//If at the top left corner then search the pixel by moving right until you reach top right corner
	if((xl < x) && (yl < y)) {
		for(;xl<=(x+dist);xl++) {
		  if(((yl > 0) && (xl > 0)) && ((yl < iBgHeight) && (xl < iBgWidth))){
			if((*image).at<Vec3b>(yl, xl)[channel] == color) {
				*offsetx += xl;
				*offsety += yl;
				rescounter++;
			}
		  }
		}
		xl = x + dist;
	}

	//If at the top right corner then search the pixel by moving down until you reach bottom right corner
	if((xl > x) && (yl < y)) {
		for(;yl<=(y+dist);yl++) {
		  if(((yl > 0) && (xl > 0)) && ((yl < iBgHeight) && (xl < iBgWidth))){
			if((*image).at<Vec3b>(yl, xl)[channel] == color) {
				*offsetx += xl;
				*offsety += yl;
				rescounter++;
			}
		  }
		}
		yl = y + dist;
	}

	//If at the bottom right corner then search the pixel by moving left until you reach botom left corner
	if((xl > x) && (yl > y)) {
		for(;xl>=(x-dist);xl--) {
		  if(((yl > 0) && (xl > 0)) && ((yl < iBgHeight) && (xl < iBgWidth))){
			if((*image).at<Vec3b>(yl, xl)[channel] == color) {
				*offsetx += xl;
				*offsety += yl;
				rescounter++;
			}
		  }
		}
		xl = x - dist;
	}

	//If at the botom left corner then search the pixel by moving up until you reach top left corner
	if((xl < x) && (yl > y)) {
		for(;yl>=(y-dist);yl--) {
		  if(((yl > 0) && (xl > 0)) && ((yl < iBgHeight) && (xl < iBgWidth))){
			if((*image).at<Vec3b>(yl, xl)[channel] == color) {
				*offsetx += xl;
				*offsety += yl;
				rescounter++;
			}
		  }
		}
		yl = y - dist;
	}

	if(rescounter == 0) {
		//If no pixels with matching color were found return zero
		return 0;
	} else {
		//Otherwise set the offset coordinates to the mean coordinates of all detected pixels and return the amount of detected pixels
		*offsetx /= rescounter;
		*offsety /= rescounter;
		return rescounter;
	}	
}

int BallDetection::countblackpixelsaroundatedge(Mat * image, int centerx, int centery, int halfsquaresize) {
	int xl, yl;
	int result=0;
	//Go through all pixels in the square and increment the result on each dark pixel
	for(xl=(centerx-halfsquaresize); xl<=(centerx+halfsquaresize); xl++) {
		for(yl=(centery-halfsquaresize); yl<=(centery+halfsquaresize); yl++) {
			if(((yl > 0) && (xl > 0)) && ((yl < iBgHeight) && (xl < iBgWidth))){
			      if((*image).at<Vec3b>(yl, xl)[2] < 64) {
				      result++;
			      }
			}
		}
	}
	//Go through all pixels in a square with a with and height less by two pixels compared to the previous square and decrement the result on each dark pixel
	for(xl=(centerx-halfsquaresize+1); xl<=(centerx+halfsquaresize-1); xl++) {
		for(yl=(centery-halfsquaresize+1); yl<=(centery+halfsquaresize-1); yl++) {
			if(((yl > 0) && (xl > 0)) && ((yl < iBgHeight) && (xl < iBgWidth))){
			      if((*image).at<Vec3b>(yl, xl)[2] < 64) {
				      result--;
			      }
			}
		}
	}
	return result;
}

void BallDetection::locateBallMovingMask(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned>>& cForegroundList) {
	int table[HALFTABLESIZE*2+1][HALFTABLESIZE*2+1];
	
	int x, y;

	//Calculate all entries of the mask
	for(x = -HALFTABLESIZE; x <= HALFTABLESIZE; x++) {
		for(y = -HALFTABLESIZE; y <= HALFTABLESIZE; y++) {
			if(((int)sqrt((double)(x*x+y*y))) <= MINRADIUS)
				table[x+HALFTABLESIZE][y+HALFTABLESIZE] = 1;
			else if(((int)sqrt((double)(x*x+y*y))) > IGNORERADIUS)
				table[x+HALFTABLESIZE][y+HALFTABLESIZE] = 0;
			else if(((int)sqrt((double)(x*x+y*y))) > MAXRADIUS)
				table[x+HALFTABLESIZE][y+HALFTABLESIZE] = -1;
			else
				table[x+HALFTABLESIZE][y+HALFTABLESIZE] = 0;
		}
	}  
  
	//Draw cocentric circles represening the mask
	Point center2(50, 50);
	circle(dstFrame, center2, MINRADIUS, Scalar(255,0,0), 1,8,0);
	circle(dstFrame, center2, MAXRADIUS, Scalar(0,0,255), 1,8,0);
	circle(dstFrame, center2, IGNORERADIUS, Scalar(0,255,0), 1,8,0);
	
	int xl, yl, pixel, counter;
	int dist;
	int chainlength;
	int maxchainlength=0;
	int maxchainstartx;
	int maxchainstarty;
	int maxchainendx;
	int maxchainendy;
	Mat blackNWhiteImage;

	//Covert the input image to grayscale and copy all the resulting pixels which have their coordinates listed inside the vector to the foreground image
	cvtColor(dstFrame, blackNWhiteImage, CV_RGB2GRAY);
	Mat foreground(iBgHeight, iBgWidth, CV_8UC1, Scalar(0.0));
	for(vector< pair<unsigned,unsigned> >::const_iterator it = cForegroundList.begin(); it != cForegroundList.end(); ++it) {
		x = (*it).first;
		y = (*it).second;
		foreground.at<uchar>(y, x) = blackNWhiteImage.at<uchar>(y, x);
	}

	//Decrement the intensity of the pixels in the history image
	for(x = 0; x < iBgWidth; x++) {
		for(y = 0; y < iBgHeight; y++) {
			if(path.at<Vec3b>(y, x)[2] > 0)
				path.at<Vec3b>(y, x)[2]--;
			/*else
				path.at<Vec3b>(y, x)[2]=0;*/
		}
	}

	//Go through the new foreground pixels again check the pixels around the listet coordinates
	for(vector< pair<unsigned,unsigned> >::const_iterator it = cForegroundList.begin(); it != cForegroundList.end(); ++it) {
		x = (*it).first;
		y = (*it).second;

		counter = 0;

		for(xl = -HALFTABLESIZE; xl <= HALFTABLESIZE; xl++) {
			for(yl = -HALFTABLESIZE; yl <= HALFTABLESIZE; yl++) {
				pixel = foreground.at<uchar>(y+yl, x+xl);
				if((table[xl+HALFTABLESIZE][yl+HALFTABLESIZE] == 1) && (pixel > HTHRESHOLD)) {
					counter++; //Increment the counter by one for each bright pixel which is supposed to be bright
				} else if((table[xl+HALFTABLESIZE][yl+HALFTABLESIZE] == -1) && (pixel > LTHRESHOLD)) {
					counter-=1024; //Decrement the counter by 1024 for each bright pixel which is not supposed to be bright
				}
			}
		}
		//If the counter is over a threshold of 5 mark the pixel in the history as a possible center of the ball with bright red
		if(counter > 5) {
			path.at<Vec3b>(y, x) = {0, 0, 255};
		}
	}

	//Copy the backround to the output imagedepending on what has been selected
	if(background == 0) {
		dstFrame = path.clone();
	} else if (background == 1) {
		dstFrame = srcFrame.clone();
		return;
	}

	//Go through all new foreground pixels again and try to find moving objects there by using the history
	for(vector< pair<unsigned,unsigned> >::const_iterator it = cForegroundList.begin(); it != cForegroundList.end(); ++it) {
		//Start the search at the position of the new foreground pixel
		x = (*it).first;
		y = (*it).second;
		//The color we are looking for will be decremented with each new step
		//Maximum amout of steps: 250 (the remaining color is almost black)
		for(chainlength = 0; chainlength < 250; chainlength++) {
			//Search for pixels with the next color around the current position
			//Begin with a radius of 1 pixel and increment the radius after each unsuccessful uttempt until the pixel is found
			//Or the limit for the radius length is reached
			for(dist = 1; dist < (IGNORERADIUS); dist++) {
				xl = x;
				yl = y;
				if(searchpixelatdistancebycolor(&path, 2, 255-1-chainlength, x, y, &xl, &yl, dist)) {
					break;
				}
			}
			if(dist == (IGNORERADIUS)) { //If the limit was reached then end the chain
				break;
			} else { //Otherwise draw a blue line from the previous pixel to the current pixel
				Point start(x, y);
				x = xl;
				y = yl;
				Point end(x, y);
				line(dstFrame, start, end, Scalar(255,30*(chainlength%6),0));
			}
		}
		//The chain of steps for the current new forground pixel ends here
		//If the amount of steps of the current chain ist bigger than the amount of steps in the biggest chain so far
		//Then concider the current chain as the chain with maximum length
		if(chainlength > maxchainlength) {
			maxchainlength = chainlength;
			maxchainstartx = (*it).first;
			maxchainstarty = (*it).second;
			maxchainendx = x;
			maxchainendy = y;
		}
	}
	//If there chain with the maximum length is longer than 6 steps then draw a white circle where it begins
	if(maxchainlength > 6) {
		Point start(maxchainstartx, maxchainstarty);
		circle(dstFrame, start, IGNORERADIUS, Scalar(255,255,255), 1,8,0);
	}	
	return;
}

void BallDetection::locateBallHistoryBased(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned>> & cForegroundList) {

	//Draw cocentric circles represening the mask
	Point center2(50, 50);
	circle(dstFrame, center2, MINRADIUS, Scalar(255,0,0), 1,8,0);
	circle(dstFrame, center2, MAXRADIUS, Scalar(0,0,255), 1,8,0);
	circle(dstFrame, center2, IGNORERADIUS, Scalar(0,255,0), 1,8,0);
	
	int x, y, xl, yl, pixel, counter;
	int dist;
	int chainlength;
	int maxchainlength=0;
	int maxchainstartx;
	int maxchainstarty;
	int maxchainendx;
	int maxchainendy;
	double firstdiffx, firstdiffy;
	double thisdiffx, thisdiffy;
	int startx, starty;	
	int endx, endy;
	int foundcnt;
	int endstartlength;
	Mat blackNWhiteImage;

	cvtColor(srcFrame, blackNWhiteImage, CV_RGB2GRAY);

	//Decrement the intensity of the pixels in the history image
	for(x = 0; x < iBgWidth; x++) {
		for(y = 0; y < iBgHeight; y++) {
			if(path.at<Vec3b>(y, x)[2] > 0)
				path.at<Vec3b>(y, x)[2]--;
			/*else
				path.at<Vec3b>(y, x)[2]=0;.*/
		}
	}

	//Mark the new forground pixels in the history with bright red
	for(vector< pair<unsigned,unsigned> >::const_iterator it = cForegroundList.begin(); it != cForegroundList.end(); ++it) {
		x = (*it).first;
		y = (*it).second;
		path.at<Vec3b>(y, x) = {0, 0, 255};
	}

	//Copy the backround to the output image depending on what has been selected
	if(background == 0) {
		dstFrame = path.clone();
	} else if (background == 1) {
		dstFrame = srcFrame.clone();
		return;
	}
	
	int xCounter = 0;
	int yCounter = 0;
	int amountOfCandidates = 0;

	//Go through the new foreground pixels again and try to find moving objects there by using the history
	for(vector< pair<unsigned,unsigned> >::const_iterator it = cForegroundList.begin(); it != cForegroundList.end(); ++it) {
		//Start the search at the position of the new foreground pixel
		x = (*it).first;
		y = (*it).second;
		startx = (*it).first;
		starty = (*it).second;
		//The color we are looking for will be decremented with each new step
		//Maximum amout of steps: 250 (the remaining color is almost black)
		for(chainlength = 0; chainlength < 250; chainlength++) {

			//Search for pixels with the next color around the current position
			//Begin with a radius of 1 pixel and increment the radius after each unsuccessful uttempt until the pixel is found
			//Or the limit for the radius length is reached
			for(dist = 1; dist < MAXSTEPDISTANCE; dist++) {
				xl = x;
				yl = y;
				foundcnt = searchpixelatdistancebycolormeanresults(&path, 2, 255-1-chainlength, x, y, &xl, &yl, dist);
				if(foundcnt > 0) {
					break;
				}
			}
			//If more than 16 pixels were found then end the chain
			if(foundcnt > 16) {
				break;
			}
			if(countblackpixelsaroundatedge(&path, xl, yl, 10) < 60) { //If there are less than 60 dark pixels in the boarder of a square with a side length of 20 around the current pixel then end the chain
				break;
			}
			if((dist == MAXSTEPDISTANCE)/*||(dist == 1)*/) { //If the limit was reached then end the chain
				break;
			} else { //Otherwise draw a blue line from the previous pixel to the current pixel
				Point start(x, y);
				x = xl;
				y = yl;
				Point end(x, y);
				line(dstFrame, start, end, Scalar(255,(250/MINCHAINLENGTH)*(chainlength % (MINCHAINLENGTH-1)),0));
			}
		}
		endx = x;
		endy = y;

		//If the chain has less steps then a sertain threshold then do not concider it as the path of the ball
		if(chainlength >= MINCHAINLENGTH) {
			//Also check the distance between the starting and ending pixels of the chain
			endstartlength = vectlength(startx-endx, starty-endy);
			if(endstartlength >= MINSTARTENDDISTANCE) { //Sum up the coordinates of all possible locations of the ball
					xCounter += startx;
					yCounter += starty;
					amountOfCandidates++;
			} else { //If the chain is not long enough then draw a yellow line from its first pixel to its last pixel
				Point start2(startx, starty);
				Point end2(endx, endy);
				line(dstFrame, start2, end2, Scalar(0,255,255));
			}
		}
	}
	
	if(amountOfCandidates!=0){ //If some possible ball locations were found then calculate the mean coordinates and draw a grren circle there
	    xCounter/=amountOfCandidates;
	    yCounter/=amountOfCandidates;
	    Point start(xCounter, yCounter);
	    circle(dstFrame, start, MAXSTEPDISTANCE, Scalar(0,255,0), 1,8,0);
	    oldBallPositions[currentInsertPosition][0] = xCounter;
	    oldBallPositions[currentInsertPosition][1] = yCounter;
	    currentInsertPosition = (++currentInsertPosition)%8;
	}
	return;
}

void BallDetection::locateBallCandidatesOld(Mat& srcFrame, Mat& dstFrame, uint32_t * candidates) {
	int x, y;

	//Draw cocentric circles represening the mask
	Point center2(50, 50);
	circle(dstFrame, center2, MINRADIUS, Scalar(255,0,0), 1,8,0);
	circle(dstFrame, center2, MAXRADIUS, Scalar(0,0,255), 1,8,0);
	circle(dstFrame, center2, IGNORERADIUS, Scalar(0,255,0), 1,8,0);
	
	int xl, yl, pixel, counter;
	int dist;
	int chainlength;
	int maxchainlength=0;
	int maxchainstartx;
	int maxchainstarty;
	int maxchainendx;
	int maxchainendy;
	Mat blackNWhiteImage;

	//Covert the input image to grayscale and copy all the resulting pixels which have their coordinates listed inside the array to the foreground image
	cvtColor(dstFrame, blackNWhiteImage, CV_RGB2GRAY);
	Mat foreground(iBgHeight, iBgWidth, CV_8UC1, Scalar(0.0));
	int i;
	for(i = 1; i < candidates[0]; i++) {
		x = UNPACKHI32(candidates[i]);
		y = UNPACKLO32(candidates[i]);
		if(x == 0 && y == 0)
			break;
		foreground.at<uchar>(y, x) = blackNWhiteImage.at<uchar>(y, x);
	}

	//Decrement the intensity of the pixels in the history image
	for(x = 0; x < iBgWidth; x++) {
		for(y = 0; y < iBgHeight; y++) {
			if(path.at<Vec3b>(y, x)[2] > 0)
				path.at<Vec3b>(y, x)[2]--;
		}
	}




	//Mark the new candidates in the history with bright red
	for(i = 1; i < candidates[0]; i++) {
		x = UNPACKHI32(candidates[i]);
		y = UNPACKLO32(candidates[i]);
		if(x == 0 && y == 0)
			break;
		counter = 0;
		path.at<Vec3b>(y, x) = {0, 0, 255};

	}

	//Copy the backround to the output image depending on what has been selected
	if(background == 0) {
		dstFrame = path.clone();
	} else if (background == 1) {
		dstFrame = srcFrame.clone();
		return;
	}


	//Go through the new candidates again and try to find moving objects there by using the history
	for(i = 1; i < candidates[0]; i++) {
		x = UNPACKHI32(candidates[i]);
		y = UNPACKLO32(candidates[i]);
		if(x == 0 && y == 0)
			break;

		//The color we are looking for will be decremented with each new step
		//Maximum amout of steps: 10
		for(chainlength = 0; chainlength < 10; chainlength++) {
			//Search for pixels with the next color around the current position
			//Begin with a radius of 2 pixel and increment the radius after each unsuccessful uttempt until the pixel is found
			//Or the limit for the radius length is reached
			for(dist = 2; dist < (IGNORERADIUS); dist++) {
				xl = x;
				yl = y;
				if(searchpixelatdistancebycolor(&path, 2, 255-1-chainlength, x, y, &xl, &yl, dist)) {
					break;
				}
			}
			if(dist == (IGNORERADIUS)) { //If the distance limit is reached then end the chain
				break;
			} else { //Otherwise draw a blue line from the previous pixel to the current pixel
				Point start(x, y);
				x = xl;
				y = yl;
				Point end(x, y);
				line(dstFrame, start, end, Scalar(255,30*(chainlength%6),0));
			}
		}

		//If there chain is not longer than 6 steps then do not concider it als a possible path of the ball
		if(chainlength > 6) {
			Point start2(UNPACKHI32(candidates[i]), UNPACKLO32(candidates[i]));
			Point end2(x, y);
			//Check the distance between the start of the chain and the end of the chain
			if(diffabs(UNPACKHI32(candidates[i]), UNPACKLO32(candidates[i]), x, y) > 64) { //If the distance is longer then 8 pixels concider it as the path of the ball
				//Draw a green line form the start of the chain to the end of the chain
				line(dstFrame, start2, end2, Scalar(0,255,0));
				//Mark the location of the ball with a yellow circle
				circle(dstFrame, start2, IGNORERADIUS, Scalar(0,255,255), 1,8,0);
			} else { //If the distance is not then 8 pixels do not concider it as the path of the ball
				//Draw a purple line form the start of the chain to the end of the chain
				line(dstFrame, start2, end2, Scalar(255,0,255));
			}
		}
	}
	return;
}


void BallDetection::clearcandidategrid(void) {
	//Clear all cells of the candidate grid
	int x, y;
	for(x=0; x<candidategrid->xcells; x++) {
		for(y=0; y<candidategrid->ycells; y++) {
			candidategrid->GetEntry(x, y)->clear();
		}
	}
}

void BallDetection::putcandidatestogrid(uint32_t * candidates) {
	int x, y;
	int i;
	//Go through all new candidates inside the array and put them to the according grid cell if they are not out of range
	for(i = 1; i < (candidates[0]); i++) {
		x = UNPACKHI32(candidates[i]);
		y = UNPACKLO32(candidates[i]);
		if(((x == 0) && (y == 0)) || ((x >= iBgWidth) || (y >= iBgHeight))) {
			break;
		} else {
			candidategrid->GetEntry(x/GRID_CELLSIZE, y/GRID_CELLSIZE)->push_front(Candidate(x, y));

		}
	}
}

void BallDetection::insertclosestcandidate(Path * p, int minradius, int maxradius) {
	Candidate * bestsofar = NULL;
	float bestdist = 1000000.0;
	Candidate * current;
	float currentdist;
	int thiscellx = p->FrontCandidate().x/GRID_CELLSIZE;
	int thiscelly = p->FrontCandidate().y/GRID_CELLSIZE;
	int othercellsx;
	int othercellsy;
	list<Candidate> * candidatelist;
	list<Candidate>::iterator it;

	//Calculate the range of the cells around the current cell we will have to check depending on the maximal distance
	int cellsizesaroundtocheck = (maxradius/GRID_CELLSIZE)+1;

	//Go through all cells which migh contain the required candidates
	for(othercellsx = thiscellx - cellsizesaroundtocheck; othercellsx <= thiscellx + cellsizesaroundtocheck; othercellsx++) {
		for(othercellsy = thiscelly - cellsizesaroundtocheck; othercellsy <= thiscelly + cellsizesaroundtocheck; othercellsy++) {
			candidatelist = candidategrid->GetEntry(othercellsx, othercellsy);
			//If the cell exists (we are not out of range of the grid) then go through all elements of the list of the entry
			if(candidatelist != NULL) {
				for (it=candidatelist->begin(); it!=candidatelist->end(); it++) {
					current = &(*it);
					//Calculate the distance between the start of the current path and the current candidate
					currentdist = diffabs(current->x, current->y, p->FrontCandidate().x, p->FrontCandidate().y);
					//Do not allow the creation of new paths beginning by using the candidates which are to close to the beginning of the current path
					if(currentdist <= maxradius*maxradius){
					      current->cantStartANewPath = true;
					}
					//If the candidate is closer then the closest candidate so far then concider is as the new closest candidate
					if((currentdist < bestdist)&&((currentdist >= minradius*minradius)&&(currentdist <= maxradius*maxradius))) {
						bestdist = currentdist;
						bestsofar = current;
					}
				}
			}
		}
	}

	if(bestsofar == NULL) { //If no new candidates were found in the range then insert the prediced candidate instead
		
		p->insertPredictionCandidate();
	} else { //If a candidate was found then insert its copy into the current path, mark the path as updated to inhibit its removal		
		p->InsertCandidate(*bestsofar);
		//Also count the amount of paths this candidate was inserted into
		bestsofar->pathcnt++;
	}
}

void BallDetection::insertclosestcandidateinallpaths(int minradius, int maxradius) {
	int x, y;
	list<Path>::iterator it;
	list<Path> * l;
	Path * current;
	int amountOfPaths = 0;

	//Go through all entries of the path grid
	for(x=0; x<pathgrid->xcells; x++) {
		for(y=0; y<pathgrid->ycells; y++) {
			//Go through all elements of the list of the entry and try to connect thiese paths with new candidates
			l = pathgrid->GetEntry(x, y);
			for (it=l->begin(); it!=l->end(); it++) {
				current = &(*it);
				insertclosestcandidate(current, minradius, maxradius);
			}
			amountOfPaths += l->size();
		}
	}
	

}

void BallDetection::insertnewpath(Candidate start) {
	//Insert a new path with a single element to the list of the according grid cell
	//Mark the path accordingly to prevent its removal during the rest of the frame
	int thiscellx = start.x/GRID_CELLSIZE;
	int thiscelly = start.y/GRID_CELLSIZE;
	
	pathgrid->GetEntry(thiscellx, thiscelly)->push_front(Path(start));
	pathgrid->GetEntry(thiscellx, thiscelly)->front().inhibitremoval = true;
}

void BallDetection::createnewpathsfromremainingcandidates(void) {
	int x, y;
	list<Candidate>::iterator it;
	list<Candidate> * l;
	Candidate current;

	//Go through all entries of the candidate grid
	for(x=0; x<candidategrid->xcells; x++){
		for(y=0; y<candidategrid->ycells; y++) {
			//Go through all elements of the list of the entry and start a new path from all unconnected candidates
			//There are some exceptions which were marked accordingly
			l = candidategrid->GetEntry(x, y);
			for (it=l->begin(); it!=l->end(); it++) {
				current = (*it);
				if(current.cantStartANewPath == false) {
					insertnewpath(current);
				}
			}
		}
	}
}

#if 0
void BallDetection::insertinclosestpath(Candidate c, int minradius, int maxradius) {
	Path * bestsofar = NULL;
	float bestdist = 1000000.0;
	Path * current;
	float currentdist;
	int thiscellx = c.x/GRID_CELLSIZE;
	int thiscelly = c.y/GRID_CELLSIZE;
	int othercellsx;
	int othercellsy;
	list<Path> * pathlist;
	list<Path>::iterator it;



	int cellsizesaroundtocheck = (maxradius/GRID_CELLSIZE)+1;

	for(othercellsx = thiscellx - cellsizesaroundtocheck; othercellsx <= thiscellx + cellsizesaroundtocheck; othercellsx++) {
		for(othercellsy = thiscelly - cellsizesaroundtocheck; othercellsy <= thiscelly + cellsizesaroundtocheck; othercellsy++) {
			pathlist = pathgrid->GetEntry(othercellsx, othercellsy);
			if(pathlist != NULL) {
				for (it=pathlist->begin(); it!=pathlist->end(); it++) {
					current = &(*it);
					currentdist = diffabs(current->FrontCandidate().x, current->FrontCandidate().y, c.x, c.y);
					if((currentdist < bestdist)&&((currentdist >= minradius)&&(currentdist <= maxradius))) {
						bestdist = currentdist;
						bestsofar = current;
					}
				}
			}
		}
	}

	if(bestsofar == NULL) {
		bestsofar = new Path();
		pathgrid->GetEntry(thiscellx, thiscelly)->push_front(*bestsofar);
		delete(bestsofar);
		bestsofar = &pathgrid->GetEntry(thiscellx, thiscelly)->front();
	}
	bestsofar->InsertCandidate(c);
	bestsofar->inhibitremoval = true;


}

void BallDetection::insertallinclosestpath(int minradius, int maxradius) {
	int x, y;
	list<Candidate>::iterator it;
	list<Candidate> * l;
	Candidate current;

	for(x=0; x<candidategrid->xcells; x++) {
		for(y=0; y<candidategrid->ycells; y++) {
			l = candidategrid->GetEntry(x, y);
			for (it=l->begin(); it!=l->end(); it++) {
				current = (*it);
				insertinclosestpath(current, minradius, maxradius);
			}
		}
	}

}
#endif

void BallDetection::removeorrelocatepaths(void) {
	int x, y;

	list<Path>::iterator it;
	list<Path> * l;
	//Path currentpath(Candidate(0,0));

	//Go through all entries of the path grid
	for(x=0; x<pathgrid->xcells; x++) {
		for(y=0; y<pathgrid->ycells; y++) {
			l = pathgrid->GetEntry(x, y);
			//Go through all elements of the list of the entry
			for (it=l->begin(); it!=l->end();) {
				//Remove the path from the list if it is not a new path and there was no new new candidte attached to it
				//Also remove the path if last attached candidate is out of the range of the frame
				int tempX = (*it).FrontCandidate().x;
				int tempY = (*it).FrontCandidate().y;
				if((*it).inhibitremoval == false || tempX < 0 || tempY < 0 || tempX > iBgWidth || tempY > iBgHeight) {
					it=l->erase(it);
					if(it == l->end())
						break;
				}else{
				   it++;
				}
			}
		}
	}

	//Go through all entries of the path grid
	for(x=0; x<pathgrid->xcells; x++) {
		for(y=0; y<pathgrid->ycells; y++) {
			l = pathgrid->GetEntry(x, y);
			//Go through all elements of the list of the entry
			for (it=l->begin(); it!=l->end(); it++) {
				//Check the coordinates of all new candidates inserted into the paths and relocate the paths if the coordinates are out of range of the grid cell
				Path currentpath = (*it);
				int tempX = currentpath.FrontCandidate().x;
				int tempY = currentpath.FrontCandidate().y;
				if(((tempX)/GRID_CELLSIZE != x) || ((tempY)/GRID_CELLSIZE != y)) {
				      if(tempX > 0 && tempY > 0 && tempX < iBgWidth && tempY < iBgHeight) {
					    pathgrid->GetEntry(currentpath.FrontCandidate().x/GRID_CELLSIZE, currentpath.FrontCandidate().y/GRID_CELLSIZE)->push_front(*it);
					    it=l->erase(it);
					    if(it == l->end())
						    break;
				      }
					
				}
				  
			}
		}
	}	
}

void BallDetection::removepathswithsamestart(void) {
	int x, y;
	int cx, cy;
	list<Candidate>::iterator cit;
	list<Candidate> * cl;
	list<Path>::iterator pit;
	list<Path> * pl;

	//Go through all entries of the candidate grid
	for(x=0; x<candidategrid->xcells; x++) {
		for(y=0; y<candidategrid->ycells; y++) {
			cl = candidategrid->GetEntry(x, y);
			//Go through all elements of the list of the entry
			for (cit=cl->begin(); cit!=cl->end(); cit++) {
				//If the amount of paths starting at the same candidate is over a certain threshold then remove all thiese paths from the path grid
				if(cit->pathcnt > MAXPATHSWITHSAMESTART) {
					cx = cit->x;
					cy = cit->y;				  
					pl = pathgrid->GetEntry(x, y);
					for (pit=pl->begin(); pit!=pl->end(); pit++) {
						if((pit->FrontCandidate().x == cx) && (pit->FrontCandidate().y == cy)){
							pit=pl->erase(pit);
							if(pit == pl->end())
								break;
						}
					}
				}
			}
		}
	}
}
  
void BallDetection::locateBallCandidatesNew(Mat& srcFrame, Mat& dstFrame, uint32_t * candidates) {
	int x, y;
	list<Path>::iterator itp;
	list<Path> * lp;
	list<Candidate>::iterator itc;
	list<Candidate> * lc;
	int i;

	clearcandidategrid(); //Clear all cells of the candidate grid
	putcandidatestogrid(candidates); //Insert all new candidates to the according grid cells
	insertclosestcandidateinallpaths(2, 16); //Try to connect the paths to the new candidates
	createnewpathsfromremainingcandidates(); //Create new paths from unconnected candidates if there were not marked otherwise
	removeorrelocatepaths(); //Remove paths with no new candidates and paths out of range
	removepathswithsamestart(); //If to many paths start at the same point remove them

	//Copy the backround to the output image depending on what has been selected
	if(background == 0) {
		dstFrame = path.clone();
	} else if (background == 1) {
		dstFrame = srcFrame.clone();
		return;
	}

	float maxLength = 0;
	Point centerOfThatPoint(0,0);

	//Go through all entries of the path-grid
	for(x=0; x<pathgrid->xcells; x++) {
		for(y=0; y<pathgrid->ycells; y++) {
			lp = pathgrid->GetEntry(x, y);
			//Go through all list elements in each entry
			for (itp=lp->begin(); itp!=lp->end(); itp++) {
				lc = &((*itp).path);
				if((*itp).ball) {
					Point start(lc->front().x, lc->front().y);
					Point end(lc->back().x, lc->back().y);		
				  
					if(abs((*itp).overallMovementX)*abs((*itp).overallMovementY) > maxLength){
					    maxLength = abs((*itp).overallMovementX)*abs((*itp).overallMovementY);
					    centerOfThatPoint = Point(lc->front().x, lc->front().y);
					    
					}
					circle(dstFrame, start, 8, Scalar(255,0,0), 1,8,0);
				}else{
				  Point start(lc->front().x, lc->front().y);
				  circle(dstFrame, start, 8, Scalar(0,0,255), 1,8,0);
				}
			}

		}
	}
	//Draw a bold green circle around the point we have found
	if(centerOfThatPoint.x != 0 && centerOfThatPoint.y != 0){
	    circle(dstFrame, centerOfThatPoint, 16, Scalar(0,255,0), 5,8,0);
	}
	

}

