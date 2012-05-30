/*
 *	File: VideoBackend.cpp
 *	---------------------------
 *	The class implementations of the video backend (disk writer / display) engine.
 *
 *	Author: Michael Andersch, 2012
 */

#include "VideoBackend.h"
#include "global.h"

void VideoBackend::setFileParams(const string& filename, int codec, double fps, Size framesize, bool isColor) {
	if(!bWriteStreamToDisk)
		return;
	vw.open(filename, codec, fps, framesize, isColor);
	if(!vw.isOpened()) {
		cerr << ERROR("could not open output file location") << endl;
		exit(EXIT_FAILURE);
	}
};

VideoBackend& VideoBackend::operator<< (Mat& img) {
	if(bWriteStreamToDisk) {
		if(!vw.isOpened()) {
			cerr << ERROR("could not write to output file location") << endl;
			exit(EXIT_FAILURE);
		}
		vw << img;
	} else {
		imshow("VideoOutput", img);
	}
	return (*this);
}