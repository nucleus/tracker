/*
 *	File: VideoBackend.h
 *	---------------------------
 *	The class definition of the video backend (disk writer / display) engine.
 *
 *	Author: Michael Andersch, 2012
 */
#ifndef BACKEND_H_
#define BACKEND_H_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cassert>

using namespace std;
using namespace cv;

/*!
 * 	Class: VideoBackend.
 * 
 * 	Class responsible for handling feedback of the processed video stream
 * 	to the user. VideoBackend is capable of handling output to the screen
 * 	and to system disk.
 */
class VideoBackend {
public:
	/*!
	 *	Constructor.
	 * 
	 *	Takes either DISK or SCREEN as output device
	 */
	VideoBackend(string device) {
		bWriteStreamToDisk = (device == "DISK");
		if(!bWriteStreamToDisk)
			namedWindow("VideoOutput", 1);
	};
	
	VideoBackend& operator<< (Mat& img);
	
	/*!
	 * 	Function: setFileParameters.
	 * 
	 * 	Configures the VideoWriter object to write the desired type of video.
	 */
	void setFileParams(const string& filename, int codec, double fps, Size framesize, bool isColor = true);
private:
	bool bWriteStreamToDisk;
	VideoWriter vw;
};

#endif
