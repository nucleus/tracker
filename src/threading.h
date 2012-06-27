/*
 * 	File: threading.h
 * 	-----------------
 * 	Function headers for CPU threading.
 * 
 * 	Written by Michael Andersch, 2012.
 */

#ifndef _THREADING_H_
#define _THREADING_H_

#include <queue>
#include <boost/thread.hpp>

// Queue class that has thread synchronisation
template <typename T>
class SynchronisedQueue {
private:
	std::queue<T> m_queue;			// Use STL queue to store data
	boost::mutex m_mutex;			// The mutex to synchronise on
	boost::condition_variable m_cond;	// The condition to wait for
	boost::condition_variable m_cond_reader;
	unsigned cur, max;
public:
	SynchronisedQueue(unsigned _max) {
		cur = 0;
		max = _max;
	}
	
	// Add data to the queue and notify others
	void Enqueue(const T& data)
	{
		// Acquire lock on the queue
		boost::unique_lock<boost::mutex> lock(m_mutex);

		while(cur >= max) m_cond.wait(lock);
		// Add the data to the queue
		m_queue.push(data);
		cur++;

		// Notify others that data is ready
		m_cond_reader.notify_one();

	} // Lock is automatically released here

	// Get data from the queue. Wait for data if not available
	T Dequeue()
	{

		// Acquire lock on the queue
		boost::unique_lock<boost::mutex> lock(m_mutex);

		// When there is no data, wait till someone fills it.
		// Lock is automatically released in the wait and obtained 
		// again after the wait
		while (cur == 0) m_cond_reader.wait(lock);
		cur--;
		// Retrieve the data from the queue
		T result=m_queue.front(); m_queue.pop();
		m_cond.notify_one();
		return result;

	} // Lock is automatically released here
};

// Thread to read in frames from the video stream
class ReaderThread {
private:
	VideoCapture* cap;
	Mat* frame;
	Mat tmp;
	SynchronisedQueue<Mat*>* Q;
	unsigned width, height;
public:
	ReaderThread(VideoCapture* _cap, SynchronisedQueue<Mat*>* queue, unsigned w, unsigned h);
	
	void operator()();
};

// Thread to process frames, could be split later into segmenter and ball detection
class ProcessorThread {
private:
	SynchronisedQueue<Mat*>* InputQueue;
	SynchronisedQueue<Mat*>* OutputQueue;
	
	bool bUseGPU, updateBackground, firstFrame;
	
	ForegroundSegmenter fg;
	BallDetection bd;
	detectionAlgorithm algo;
	
	unsigned processedFrames, maxFrames;
public:
	ProcessorThread(SynchronisedQueue<Mat*>* input, SynchronisedQueue<Mat*>* output, bool useGPU, unsigned width, unsigned height, unsigned maxframes, double lrate, detectionAlgorithm _algo);
	~ProcessorThread() {
		Mat bg = fg.modelMean();
		imwrite("bgmodel.jpg", bg);
	}

	void operator()();
};

#endif