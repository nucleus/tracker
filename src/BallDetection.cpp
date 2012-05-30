#include "BallDetection.h"
#include "TrackerExceptions.h"

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
	
	tracking = false;
	
// 	int x, y;
// 	for(x = -HALFTABLESIZE; x <= HALFTABLESIZE; x++) {
// 		printf("\n");
// 		for(y = -HALFTABLESIZE; y <= HALFTABLESIZE; y++) {
// 			if(((int)sqrt((double)(x*x+y*y))) <= MINRADIUS)
// 				table[x+HALFTABLESIZE][y+HALFTABLESIZE] = 1;
// 			else if(((int)sqrt((double)(x*x+y*y))) > IGNORERADIUS)
// 				table[x+HALFTABLESIZE][y+HALFTABLESIZE] = 0;
// 			else if(((int)sqrt((double)(x*x+y*y))) > MAXRADIUS)
// 				table[x+HALFTABLESIZE][y+HALFTABLESIZE] = -1;
// 			else
// 				table[x+HALFTABLESIZE][y+HALFTABLESIZE] = 0;
// 		}
// 	}
};

void BallDetection::setImageParams(unsigned int _width, unsigned int _height, unsigned int _channels) {
	iBgWidth = _width;
	iBgHeight = _height;
	assert(_channels == 1);
	iBgChannels = _channels;
	path = Mat(iBgHeight, iBgWidth, CV_8UC3, Scalar(0.0)).clone();
	HALFTABLESIZE = 7; 
	MINRADIUS = 3;
	MAXRADIUS = 5;
	IGNORERADIUS = 15;	
}

void BallDetection::searchBall(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned>>& cForegroundList, detectionAlgorithm algo){
	switch(algo) {
		case ALGO_MOVING:
			locateBallMovingMask(dstFrame, dstFrame, cForegroundList);
			break;
		case ALGO_CLUSTER:
			locateBallForegroundClusters(srcFrame, dstFrame, cForegroundList);
			break;
		case ALGO_OPTICAL:
			locateBallOpticalFlow(srcFrame, dstFrame, cForegroundList);
			break;
		default:
			throw IllegalArgumentException();
	}
}

void BallDetection::locateBallOpticalFlow(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned>>& cForegroundList) {
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
		float traveledDistance = sqrt((it->x-itnext->x)*(it->x-itnext->x) + (it->y-itnext->y)*(it->y-itnext->y));
		distances.push_back(traveledDistance);
// 		cout << traveledDistance << endl;
		if(traveledDistance > 0.6f) {
			count++;
			currentFilteredPoints.push_back(Point2f(it->x, it->y));
			nextFilteredPoints.push_back(Point2f(itnext->x, itnext->y));
		}
		itnext++;
	}

	bool detected = (count > BALL_PELS_LOWER && count < BALL_PELS_UPPER);

	// If ball was detected with reasonable accuracy, draw it and add current position to past ball positions
	if(detected) {
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
		if(!tracking) {
			vPreviousBallLocations.clear();
			tracking = true;
		}
			
		// Find ball corner, find minimum in both x and y direction
		float x, y;
		sort(currentFilteredPoints.begin(), currentFilteredPoints.end(), cmpPointPairX);
		x = currentFilteredPoints[9].x;
		sort(currentFilteredPoints.begin(), currentFilteredPoints.end(), cmpPointPairY);
		y = currentFilteredPoints[9].y;
// 		cout << "Located at " << x << "," << y << endl;
		vPreviousBallLocations.push_front(Point2f(x,y));
	} else {
		// If no good ball candidate, draw prediction from past candidates and delete past candidates
		tracking = false;
		
		// Linear interpolation
		if(vPreviousBallLocations.size() >= 2) {
			list<Point2f>::iterator it = vPreviousBallLocations.begin();
			Point2f last = *it++;
			Point2f secondToLast = *it;
			Point2f nextPoint;
			nextPoint.x = secondToLast.x + (last.x - secondToLast.x)*(2);
			nextPoint.y = secondToLast.y + (last.y - secondToLast.y)*(2);
			vPreviousBallLocations.push_front(nextPoint);
// 			cout << "Next point " << nextPoint.x << "," << nextPoint.y << " from " << last.x << "," << last.y << " and " << secondToLast.x << "," << secondToLast.y << endl;
		}
	}
	
	// Draw flight contours
	if(vPreviousBallLocations.size() >= 2) {
		list<Point2f>::iterator nextit, tmp;
		for(list<Point2f>::iterator it = vPreviousBallLocations.begin(); it != vPreviousBallLocations.end(); it++) {
			tmp = it;
			nextit = it++;
			it = tmp;
			if(nextit == vPreviousBallLocations.end())
				break;
			line(dstFrame, *it, *nextit, Scalar(0,0,255), 1, 8);				
		}
	}
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

//equi time distanced
vector<pair<unsigned, unsigned>> pastCoordinatesOfRecognizedBalls;

void BallDetection::addCoordinatesToVector(int recognizedX, int recognizedY){
    if(pastCoordinatesOfRecognizedBalls.size() > 15){
     pastCoordinatesOfRecognizedBalls.erase(pastCoordinatesOfRecognizedBalls.begin()); 
    }
    //insert at last position
    pastCoordinatesOfRecognizedBalls.push_back(make_pair(recognizedX, recognizedY));
    cout << "Added a new position" << endl;
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
	  cout << "Distance for closest clustercenter: " << distance[smallestCandidate]  << endl;
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
     * 
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

static int searchpixelatdistancebycolor(Mat * image, int channel, int color, int x, int y, int * offsetx, int * offsety, int dist) {
	int xl = *offsetx;
	int yl = *offsety;

	int startx = *offsetx;
	int starty = *offsety;

	if((startx == x) && (starty == y)) {
		xl = x - dist;
		yl = y - dist;
	} else {
		xl = startx;
		yl = starty;
	}

	if((xl < x) && (yl < y)) {
		for(;xl<=(x+dist);xl++) {
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

	if((xl > x) && (yl < y)) {
		for(;yl<=(y+dist);yl++) {
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

	if((xl > x) && (yl > y)) {
		for(;xl>=(x-dist);xl--) {
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

	if((xl < x) && (yl > y)) {
		for(;yl>=(y-dist);yl--) {
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

	return 0;
}


void BallDetection::locateBallMovingMask(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned>>& cForegroundList) {
	int table[HALFTABLESIZE*2+1][HALFTABLESIZE*2+1];

	
	lastImage = NULL;
	countframes = 0;

	iNumberOfClusters = 6;
	amountOfPastCenterCoords = 5;
	
	int x, y;
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
  
	Point center2(50, 50);
	circle(dstFrame, center2, MINRADIUS, Scalar(255,0,0), 1,8,0);
	circle(dstFrame, center2, MAXRADIUS, Scalar(0,0,255), 1,8,0);
	circle(dstFrame, center2, IGNORERADIUS, Scalar(0,255,0), 1,8,0);
	
	//int x, y, 
	int xl, yl, 	pixel, counter;
	int dist;
	int chainlength;
	int maxchainlength=0;
	int maxchainstartx;
	int maxchainstarty;
	int maxchainendx;
	int maxchainendy;
	Mat blackNWhiteImage;

	Mat pcopy(iBgHeight, iBgWidth, CV_8UC3, Scalar(0.0));

	cvtColor(srcFrame, blackNWhiteImage, CV_RGB2GRAY);

	Mat foreground(iBgHeight, iBgWidth, CV_8UC1, Scalar(0.0));

	for(vector< pair<unsigned,unsigned> >::const_iterator it = cForegroundList.begin(); it != cForegroundList.end(); ++it) {
		x = (*it).first;
		y = (*it).second;
		foreground.at<uchar>(y, x) = blackNWhiteImage.at<uchar>(y, x);
	}


	for(x = 0; x < iBgWidth; x++) {
		for(y = 0; y < iBgHeight; y++) {
			if(path.at<Vec3b>(y, x)[2] > 0)
				path.at<Vec3b>(y, x)[2]--;
			else
				path.at<Vec3b>(y, x)[2]=0;
		}
	}


	for(vector< pair<unsigned,unsigned> >::const_iterator it = cForegroundList.begin(); it != cForegroundList.end(); ++it) {
		x = (*it).first;
		y = (*it).second;

		counter = 0;

		for(xl = -HALFTABLESIZE; xl <= HALFTABLESIZE; xl++) {
			for(yl = -HALFTABLESIZE; yl <= HALFTABLESIZE; yl++) {
				pixel = foreground.at<uchar>(y+yl, x+xl);
				if((table[xl+HALFTABLESIZE][yl+HALFTABLESIZE] == 1) && (pixel > HTHRESHOLD)) {
					counter++;
				} else if((table[xl+HALFTABLESIZE][yl+HALFTABLESIZE] == -1) && (pixel > LTHRESHOLD)) {
					counter-=1024;
				}
			}
		}
		if(counter > 5) {
			//Point center(x, y);
			//circle(path, center, MAXRADIUS, Scalar(0,0,255), 1,8,0);
			path.at<Vec3b>(y, x) = {0, 0, 255};
			//return;
		}
	}

	pcopy = path.clone();

	dstFrame = path.clone();

	

	for(vector< pair<unsigned,unsigned> >::const_iterator it = cForegroundList.begin(); it != cForegroundList.end(); ++it) {
//		vector< pair<unsigned,unsigned> >::const_iterator it = cForegroundList.begin();
		x = (*it).first;
		y = (*it).second;
		for(chainlength = 0; chainlength < 250; chainlength++) {

			/*printf("chain is at %d %d \n", x, y);
			fflush(stdout);*/

			for(dist = 1; dist < (IGNORERADIUS); dist++) {
				xl = x;
				yl = y;
				if(searchpixelatdistancebycolor(&path, 2, 255-1-chainlength, x, y, &xl, &yl, dist)) {
					break;
				}
			}
			if(dist == (IGNORERADIUS)) {
				break;
			} else {
				Point start(x, y);
				x = xl;
				y = yl;
				Point end(x, y);
				//line(pcopy, start, end, Scalar(0,255,0));
				line(dstFrame, start, end, Scalar(255,30*(chainlength%6),0));
			}
		}
		if(chainlength > maxchainlength) {
			maxchainlength = chainlength;
			maxchainstartx = (*it).first;
			maxchainstarty = (*it).second;
			maxchainendx = x;
			maxchainendy = y;
		}
	}

	/*if(maxchainlength > 0) {
		x = maxchainstartx;
		y = maxchainstarty;
		for(chainlength = 0; chainlength < 250; chainlength++) {
			for(dist = 1; dist < (IGNORERADIUS); dist++) {
				xl = x;
				yl = y;
				if(searchpixelatdistancebycolor(&path, 2, 255-1-chainlength, x, y, &xl, &yl, dist)) {
					break;
				}
			}
			if(dist == (IGNORERADIUS)) {
				break;
			} else {
				Point start(x, y);
				x = xl;
				y = yl;
				Point end(x, y);
				//line(pcopy, start, end, Scalar(0,255,0));
				line(dstFrame, start, end, Scalar(0,255,0));
			}
		}*/
		if(maxchainlength > 6) {
			Point start(maxchainstartx, maxchainstarty);
			//circle(pcopy, start, IGNORERADIUS, Scalar(0,255,255), 1,8,0);
			circle(dstFrame, start, IGNORERADIUS, Scalar(255,255,255), 1,8,0);
			
			int iWindowForOtherBalls = 50;
	
			int ballsFoundInVincinity = 0;
			int iAmountOfFoundBallsCloseTogether();

		      
			//search for other found balls in this vincinity
			for(int iRowAround = maxchainstarty-iWindowForOtherBalls; iRowAround < maxchainstarty+iWindowForOtherBalls; iRowAround++){
			  for(int iColAround = maxchainstartx-iWindowForOtherBalls;iColAround < maxchainstartx+iWindowForOtherBalls; iColAround++){
			    if(path.at<Vec3b>(iColAround, iRowAround)[0] > 0){
				  ballsFoundInVincinity++;
				    //cout << "other balls in vincinity: " <<  ballsFoundInVincinity << endl;
			    }
			  }
			}
			if(ballsFoundInVincinity > 15){
			  if(HALFTABLESIZE>5){
			    //HALFTABLESIZE--;
			    //MINRADIUS--;
			    //MAXRADIUS--;
// 			    IGNORERADIUS--;
			    HALFTABLESIZE/=1.01;
			    MINRADIUS/=1.01;
			    MAXRADIUS/=1.01;
			    //IGNORERADIUS/=1.0001;
// 			    cout << "become smaller" << endl;
			  }
			}else if(ballsFoundInVincinity <2){
			  if(HALFTABLESIZE<20){
			    HALFTABLESIZE*=1.1; 
			    MINRADIUS*=1.1;
			    MAXRADIUS*=1.1;
// 			    IGNORERADIUS*=1.1;
// 			    cout << "become bigger" << endl;
			  }
			}
		    //cout << "other balls in vincinity: " <<  ballsFoundInVincinity << endl;
			
		}
	//}

	dstFrame:// = pcopy;
	return;
}

