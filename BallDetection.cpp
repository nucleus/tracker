#include "BallDetection.h"
#include "TrackerExceptions.h"

BallDetection::BallDetection() {
	
	lastImage = NULL;
	countframes = 0;

	iNumberOfClusters = 6;
	amountOfPastCenterCoords = 5;
	
	int x, y;
	for(x = -HALFTABLESIZE; x <= HALFTABLESIZE; x++) {
		printf("\n");
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
};

void BallDetection::setImageParams(unsigned int _width, unsigned int _height, unsigned int _channels) {
	iBgWidth = _width;
	iBgHeight = _height;
	assert(_channels == 1);
	iBgChannels = _channels;
	
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

	if(count < BALL_PELS_UPPER && count > BALL_PELS_LOWER) {
		itnext = nextFilteredPoints.begin();
		bool drawLine = true;
		for(vector<Point2f>::iterator it = currentFilteredPoints.begin(); it != currentFilteredPoints.end(); it++) {
			if(drawLine)
				line(dstFrame, *it, *itnext, Scalar(0,0,255), 1, 8);
			drawLine = !drawLine;
			itnext++;
		}
	}
	return;
}

void BallDetection::locateBallForegroundClusters(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned>>& cForegroundList) {
	int iCounter = 0;
	Mat xyMatrix = Mat::zeros(iBgHeight*iBgWidth, 2, CV_32F);
  
	for(vector< pair<unsigned,unsigned> >::const_iterator it = cForegroundList.begin(); it != cForegroundList.end(); ++it) {
		xyMatrix.at<float>(iCounter, 1) = (*it).first;
		xyMatrix.at<float>(iCounter, 0) = (*it).second;
		iCounter++;
	}

	double dEpsilon = 1.0; 
	int iAmountOfIterations = 50;		//amount of iterations of kmeans
	int iAmountOfAttemps = 2; 		//amount of new cluster init points
  
	TermCriteria termCriteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, iAmountOfIterations, dEpsilon);

	Mat mCoordinatesOfCenters = Mat::zeros(iNumberOfClusters, 2, CV_32FC1);
	Mat mLabels(xyMatrix.rows, 1, CV_32FC1);

	kmeans(xyMatrix, iNumberOfClusters, mLabels, termCriteria, iAmountOfAttemps, KMEANS_PP_CENTERS, mCoordinatesOfCenters);

	if(pastCenterCoordinates.size() > 5){
		pastCenterCoordinates.erase(pastCenterCoordinates.begin(), pastCenterCoordinates.begin());
	}
     
	iCounter = 0;
  
	int sizeOfCluster[iNumberOfClusters];
	for(int i = 0; i < iNumberOfClusters; i++){
		sizeOfCluster[i] = 0;
	}
  
  //paint each cluster in different colour
  //and calculate cluster size (amount of pixel) and get their dimensions
	for(vector< pair<unsigned,unsigned> >::const_iterator it = cForegroundList.begin(); it != cForegroundList.end(); ++it) {
		int iTempLabelNumber = mLabels.at<uchar>(iCounter,0);
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
		/*
		dstFrame.at<Vec3b>((*it).second,(*it).first)[0] = iColor[0];
		dstFrame.at<Vec3b>((*it).second,(*it).first)[1] = iColor[1];
		dstFrame.at<Vec3b>((*it).second,(*it).first)[2] = iColor[2];
		*/
		iCounter++;
	}
  //printf("Cluster 1: %i, 2 %i, 3 %i, 4 %i, 5 %i, 6 %i \r\n", sizeOfCluster[0], sizeOfCluster[1], sizeOfCluster[2], sizeOfCluster[3], sizeOfCluster[4] , sizeOfCluster[5]);
    
	//paint cluster center
	for(int i = 0 ; i < iNumberOfClusters ; i++){ 
		double dMinDistance = 1024;
		double dCurrentTempX = mCoordinatesOfCenters.at<float>(i,1);
		double dCurrentTempY = mCoordinatesOfCenters.at<float>(i,0);
		Point center(dCurrentTempX,dCurrentTempY);
		
		for(vector<Mat>::const_iterator it = pastCenterCoordinates.begin(); it != pastCenterCoordinates.end(); ++it){
			Mat tempMat = *it;
			for(int j = 0; j < tempMat.rows;j++){
				double dTempX = tempMat.at<float>(j,1);
				double dTempY = tempMat.at<float>(j,0);
				double distance = sqrt((dTempX-dCurrentTempX)*(dTempX-dCurrentTempX) + (dTempY-dCurrentTempY)*(dTempY-dCurrentTempY));
				if(distance < dMinDistance){
					dMinDistance = distance;
				}
			}
		}
		//printf("minDistance: %f \n\r ", dMinDistance);
		//lowest distance of last frames to the closest center in that time
		//distance <= 5 means, that the old center is to close to the new one, its probably the same and not moving enough

		if(dMinDistance < 1.5 || dMinDistance > 10){
			//circle(dstFrame, center, 5, Scalar(255,0,0), 1,5,0);
		}else{
			//printf("size of displayed clustercenter: %i \n\r", sizeOfCluster[i]);
			circle(dstFrame, center, 5, Scalar(0,255,0), 2,4,0);
			if(sizeOfCluster[i] <=-1 || sizeOfCluster[i] >= 200){
				//circle(dstFrame, center, 5, Scalar(0,0,255), 1,5,0);
			}else{
				//circle(dstFrame, center, 5, Scalar(0,255,0), 2,4,0);
			}
		}
	}
	pastCenterCoordinates.push_back(mCoordinatesOfCenters);
	return;
}

void BallDetection::locateBallMovingMask(Mat& srcFrame, Mat& dstFrame, vector< pair<unsigned, unsigned>>& cForegroundList) {
	Vec3b blue = {255, 0, 0};

	Point center2(50, 50);
	circle(dstFrame, center2, MINRADIUS, Scalar(255,0,0), 1,8,0);
	circle(dstFrame, center2, MAXRADIUS, Scalar(0,0,255), 1,8,0);
	circle(dstFrame, center2, IGNORERADIUS, Scalar(0,255,0), 1,8,0);
	
	int x, y, xl, yl, pixel, counter, found, foundx, foundy, skip/*, counter1, counter2, counter3*/;
	Mat blackNWhiteImage;

	cvtColor(srcFrame, blackNWhiteImage, CV_RGB2GRAY);
	Mat foreground(iBgHeight, iBgWidth, CV_8UC1, Scalar(0.0));

	for(vector< pair<unsigned,unsigned> >::const_iterator it = cForegroundList.begin(); it != cForegroundList.end(); ++it) {
		x = (*it).first;
		y = (*it).second;
		foreground.at<uchar>(y, x) = blackNWhiteImage.at<uchar>(y, x);
		dstFrame.at<Vec3b>(y, x) = blue;
		/*Point center(x, y);
		circle(dstFrame, center, MAXRADIUS, Scalar(0,0,255), 1,8,0);*/
	}
	
	cout << "Copied input pixels" << flush << endl;

	found = 0;

	for(x = (HALFTABLESIZE+50); x < (iBgWidth-HALFTABLESIZE-50); x++) {
		for(y = HALFTABLESIZE+50; y < iBgHeight-HALFTABLESIZE-50; y++) {

			counter = 0;

			for(xl = -HALFTABLESIZE; xl <= HALFTABLESIZE; xl++) {
				for(yl = -HALFTABLESIZE; yl <= HALFTABLESIZE; yl++) {
					pixel = foreground.at<uchar>(y+yl, x+xl);
					if((table[xl+HALFTABLESIZE][yl+HALFTABLESIZE] == 1) && (pixel > HTHRESHOLD))
						counter++;
					else if((table[xl+HALFTABLESIZE][yl+HALFTABLESIZE] == -1) && (pixel > LTHRESHOLD))
						counter-=10;
				}
			}
			if(counter > 5) {
				Point center(x, y);
				circle(dstFrame, center, MAXRADIUS, Scalar(0,0,255), 1,8,0);
				return;
			}			

		}
	}
	return;
}

