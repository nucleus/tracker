#ifndef EXCEPTION_H_
#define EXCEPTION_H_

#include <exception>

/*!
 * 	Class: IllegalArgumentException.
 * 
 * 	An exception for handling bad arguments. This class is empty,
 * 	it is just used for signaling that the exception occurred.
 */
class IllegalArgumentException : public std::exception {

};

#endif