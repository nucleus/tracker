/*
 *	File: Path.h
 *	---------------------------
 *	The class definition of the foreground segmentation engine.
 *
 *	Author: David laqua, Viktor, 2012
 */

#ifndef PATH_H
#define PATH_H

#include <list>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
using namespace cv;
using namespace std; 

#include "Candidate.h"

class Path {
	public:
		Path(Candidate startValue);
		list<Candidate> path;
		bool inhibitremoval;	
		
		
		bool useAcc;
		int iNotFoundMeasurements;
		
		//Kalmanfilter for this path
		KalmanFilter kalmanFilter;
		//2 measurement data are taken (x and y position)
		Mat_<float> mMeasurement; 
		//contains the current state of the kalman filter
		Mat_<float> state;
		
		//Could be a Ball
		bool ball;
		
		
		/*!	Function: InsertCandidate.
		* 
		*	insert a candidate when having a measurement and updates the kalmanfilter
		*	c = the canditade which will be added
		*/		
		void InsertCandidate(Candidate c);
		
		/*!	Function: InsertCandidate.
		* 
		*	insert a predicting, when not having a measurement, thus adding the prediction of kalmanfilter
		*/				
		void insertPredictionCandidate();
		
		
		/*!	Function: FrontCandidate.
		* 
		*	just returns first candidate of this list
		*  	return = first candidate, meaning candidate of this frame
		*/						
		Candidate FrontCandidate(void);

		/*!	Function: checkBallContrains.
		* 
		*	check contrains which have to be fullfilled in order to be shown as a ball
		* 	sets ball = true in case the constrains are ok
		*/
		void checkBallContrains();
		
		/*!	Function: deletePathNextFrame.
		 * 
		*	sets inhibitremoval = true, so that the path will be deleted in next Frame
		*/
		void deletePathNextFrame();		
		
		//containing ballinformation
		float overallMovementX;
		float overallMovementY;		
		float counterDirectionChangesX;
		float counterDirectionChangesY;	

	private:

		/*!	Function: initKalmanFilter.
		 * 
		* 	Initialisation of the kalmanfilter with x and y position 
		* @useAcc = specify whether to use velocity and acceleration or only velocity
		* @initPosX = specify X coordinate
		* @initPosY = specify Y coordinate
		*/
		void initKalmanFilter(int initPosX, int initPosY);
		
		/*!	Function: applyKalmanFilter.
		 * 
		* 	Applies the kalmanfilter to the current measurement (if its available, otherwise use only the prediction)
		* @newX = new (measured) X coordinates
		* @newY = new (measured) Y coordinates
		* @measurementAvailable = if no ball was found in this timestep
		* @return the corrected predicted/measured point
		*/
		Point applyKalmanFilter(int newX, int newY, bool measurementAvailable);
		
		/*!	Function: updateMovement.
		 * 
		* 	updates the current ball information for the constrains
		*/
		void updateMovement();		

		//counter for measurement/prediction ratio
		float counterMeasured;
		float counterPredicted;

	
		
		
};

#endif
