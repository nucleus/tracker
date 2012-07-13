#include "Candidate.h"

#include "Path.h"

Path::Path(Candidate startValue) {
	inhibitremoval = true;
	iNotFoundMeasurements = 0;
	ball = false;
	
	Mat_<float> measure(2,1);	//dummy
	
	if(useAcc){
	      Mat_<float> tempState(6,1);
	      state = tempState.clone();
	}else{
	      Mat_<float> tempState(4,1);  
	      state = tempState.clone();
	}
	
	mMeasurement = measure.clone();
	useAcc = false;
	counterMeasured = 0;
	counterPredicted = 0;
	    
	counterDirectionChangesX=0;
	counterDirectionChangesY=0;
    
	
	initKalmanFilter(startValue.x,startValue.y);
};

void Path::InsertCandidate(Candidate c) {
	updateMovement();
  
	path.push_front(c);
	inhibitremoval = true;
	
	//a measurement was possible, now apply Kalman Filter (denoise point isnt currently used)
	Point dummy = applyKalmanFilter(c.x, c.y, true);
	
	//reset Counter
	iNotFoundMeasurements = 0;
	counterMeasured++;
	checkBallContrains();
}

void Path::insertPredictionCandidate(){
	Point prediction = applyKalmanFilter(0,0,false);

	updateMovement();
	
	path.push_front(Candidate(prediction.x, prediction.y));
	
	//used as a criteria for discarding pathes
	iNotFoundMeasurements++;  
	
	//used as a criteria to show this path as a ball
	counterPredicted++;
	checkBallContrains();
}

Candidate Path::FrontCandidate(void) {
	return path.front();
}

void Path::deletePathNextFrame(){
	inhibitremoval = false;
}

void Path::updateMovement(){
      if((overallMovementX*(path.front().x - path.back().x))<0){
	  counterDirectionChangesX++;
      }
      if((overallMovementY*(path.front().y - path.back().y))<0){
	  counterDirectionChangesY++;
      }
      overallMovementX = path.front().x - path.back().x;
      overallMovementY = path.front().y - path.back().y;
}


void Path::checkBallContrains(){
      bool checkResult = true;
      int minPathSizeRequiredForCheck = 2;
      
      if(path.size()>=minPathSizeRequiredForCheck){
	  
	    //printf("%f, %f and the ratio is: ratio: %f at posX %i, posY %i \n\r",overallMovementX,overallMovementY,abs(overallMovementX/overallMovementY),FrontCandidate().x,FrontCandidate().y );
	    
	    //movement ratio comparison
	    int movementThreshold = 100;
	    int requiredPathSize = 10;
	    int amountOfPredictsBeforeDeletion = 15;
	    if(counterDirectionChangesX > 1){
		  checkResult = false;
		  //printf("counter of Dir ChangesX failed \n\r %f", counterDirectionChangesX);
		  deletePathNextFrame();
		  return;
	    }
	    if(counterDirectionChangesY > 10){
		  checkResult = false;
		  //printf("counter of Dir ChangesY failed \n\r %f" ,counterDirectionChangesY);
		  deletePathNextFrame();
		  return;
	    }
	    if(iNotFoundMeasurements > amountOfPredictsBeforeDeletion){
		  checkResult = false;
		  deletePathNextFrame();
		  return;
	    }
	    if(overallMovementY != 0){
		if(abs(overallMovementX/(float)overallMovementY) < 2){// || meanMovementX < 30){
		      //printf("movement ratio constrained failed %f, %f\n\r", overallMovementX, overallMovementY);
		      ball = false;  
		      return;
		}
	    }
	    if(abs(overallMovementX*overallMovementY) <movementThreshold){
		  ball = false;
		  return;
		  //printf("to less motion: %f \n\r", overallMovementX*overallMovementY);
		  //measurement/prediction ratio comparison
	    }
	    if(counterPredicted/(float)counterMeasured > 1){
		  //printf("measured/predicted ratio constrained failed \n\r");
		  ball = false;
		  return;
		  //printf("Dircetion ChangesX: %f, ChangesY: %f \n\r", counterDirectionChangesX,counterDirectionChangesY);  
	    }
	
	    if(path.size() < requiredPathSize){
		  checkResult = false;
		  return;
	    } 
	    if(checkResult){
		printf("Movement ratio %f\n\r", overallMovementX/(float)overallMovementY);
		printf("Motion: %f \n\r", abs(overallMovementX*overallMovementY));
		printf("Measured/Predicted ratio %f\n\r", counterPredicted/(float)counterMeasured);
		printf("DirectionChanges X: %f, Y: %f \n\r", counterDirectionChangesX,counterDirectionChangesY);
		
	    }
	  

      }
      ball = checkResult;
      //return checkResult;
}


void Path::initKalmanFilter(int initPosX, int initPosY){
  int m;
  int n = 2;	//measurement dimension = 2; posX and posY
  
  if(useAcc){
	//higher dimensionality for transition Matrix in case of "with" acceleration
	m = 6;		//dimension of prediction vector (col, row, velX, velY, accX, accY
	kalmanFilter.init(m,n,0);
	
	/*specification of transition matrix;
	* 1  0  1  0  0.5  0
	* 0  1  0  1   0   0.5
	* 0  0  1  0   1   0
	* 0  0  0  1   0   1
	* 0  0  0  0   1   0 
	* 0  0  0  0   0   1
	*/
	
	kalmanFilter.transitionMatrix = *(Mat_<float>(m, m) << 1,0,1,0,0.5,0, 	0,1,0,1,0,0.5,	0,0,1,0,1,0, 	0,0,0,1,0,1,	0,0,0,0,1,0, 	0,0,0,0,0,1);

	//initialize Kalmanfilter value
	kalmanFilter.statePost.at<float>(0) = (float)initPosX;
	kalmanFilter.statePost.at<float>(1) = (float)initPosY;
	kalmanFilter.statePost.at<float>(2) = 0;	//velocity in x direction
	kalmanFilter.statePost.at<float>(3) = 0;	//velocity in y direction
	kalmanFilter.statePost.at<float>(4) = 0;	//acceleration in x direction
	kalmanFilter.statePost.at<float>(5) = 0;	//acceletadior in y direction
	
	//initialize covariance matrixes
	setIdentity(kalmanFilter.measurementMatrix);
	setIdentity(kalmanFilter.processNoiseCov, Scalar::all(1e-4));
	setIdentity(kalmanFilter.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(kalmanFilter.errorCovPost, Scalar::all(.1));	
	   
	
	
    }else{
	//higher dimensionality for transition Matrix in case of "with" acceleration
	m = 4;		//dimension of prediction vector (col, row, velX, velY
	kalmanFilter.init(m,n,0);
	/*specification of transition matrix;
	* 1  0  1  0  
	* 0  1  0  1  
	* 0  0  1  0  
	* 0  0  0  1 
	*/
	kalmanFilter.transitionMatrix = *(Mat_<float>(m, m) << 1,0,1,0, 	0,1,0,1,	0,0,1,0, 	0,0,0,1);
	
	//initialize Kalmanfilter value
	kalmanFilter.statePost.at<float>(0) = (float)initPosX;
	kalmanFilter.statePost.at<float>(1) = (float)initPosY;
	kalmanFilter.statePost.at<float>(2) = 0;	//velocity in x direction
	kalmanFilter.statePost.at<float>(3) = 0;	//velocity in y direction
	
	//initialize covariance matrixes
	setIdentity(kalmanFilter.measurementMatrix);
	setIdentity(kalmanFilter.processNoiseCov, Scalar::all(1e-4));
	setIdentity(kalmanFilter.measurementNoiseCov, Scalar::all(1e-2));
	setIdentity(kalmanFilter.errorCovPost, Scalar::all(.1));	
    }   
    //add to the list of points of the paths
    path.push_front(Candidate(initPosX, initPosY));
}


Point Path::applyKalmanFilter(int newX, int newY, bool measurementAvailable){
      
      Mat prediction = kalmanFilter.predict();
      Point predictPoint(prediction.at<float>(0), prediction.at<float>(1));      
      
      //make a measurement
      
      if(measurementAvailable){
	  
	  mMeasurement(0) = (float)newX;
	  mMeasurement(1) = (float)newY;
	  //Mat estimated = kalmanFilter.correct(mMeasurement); 
      //Point resultPoint((int)estimated.at<float>(0),(int)estimated.at<float>(1));
      }else{	
	
	  //printf("pred: %i, pred: %i \n\r",(int)prediction.at<float>(0), (int)prediction.at<float>(1));
	  mMeasurement(0) = prediction.at<float>(0);
	  mMeasurement(1) = prediction.at<float>(1);
      }
      
      //update of our matrizes
      state = kalmanFilter.correct(mMeasurement);
      Point resultPoint((int)state.at<float>(0),(int)state.at<float>(1));
   
      return resultPoint;  
}

