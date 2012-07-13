/*	File: OptionParser.h
 * 	--------------------
 * 	Contains the class definitions of the option parser object
 * 	which encapsulates the boost program options.
 */

#ifndef _OPTPARSER_H_
#define _OPTPARSER_H_

#include <boost/program_options.hpp>
using namespace boost::program_options;

#include <iostream>
#include <fstream>
#include <iterator>
using namespace std;

#include "global.h"
#include "BallDetection.h"

/*!
 * 	Class: OptionParser.
 * 
 * 	Class encapsulating a boost library program_options option parser.
 */
class OptionParser {
public:
	OptionParser() : config("Configuration") {
		config.add_options()
		("help,h", "print help text")
		("source,s", value<string>()->default_value("DISK"), "video source (DISK/CAM)")
		("destination,d", value<string>()->default_value("SCREEN"), "video destination (DISK/SCREEN)")
		("infile,i", value<string>()->default_value("test.avi"), "input video file")
		("outfile,o", value<string>()->default_value("out.avi"), "output video file")
		("modelframes,f", value<unsigned>()->default_value(DEFAULT_MODEL_TRAINING_COUNT), "background model frames")
		("learnrate,l", value<double>()->default_value(DEFAULT_LEARNING_RATE), "background model learning rate")
		("detection,a", value<int>()->default_value(ALGO_CIRCLES), "detection algorithm [0-motion estimation, 1-clustering, 2-generalized hough transform, 3-path based, 4-grid based circles, 5-grid based kalman]")
		("gpu,g", "use GPU for processing")
		("threaded,t", "use software pipelining");
		
	};
	
	/*!
	 * 	Function: parse.
	 * 
	 * 	Uses the preconfigured and initialized option parser to parse a config file.
	 */
	void parse(int ac, char** av, string config_file) {
		ifstream ifs(config_file.c_str());
		if (!ifs) {
			cerr << "Cannot open config file: " << config_file << endl;
			exit(EXIT_FAILURE);
		} else {
			store(parse_config_file(ifs, config), vm);
			notify(vm);
		}
		
		store(command_line_parser(ac, av).options(config).run(), vm);
		notify(vm);
	};
	
	/*!
	 * 	Function: getOptions.
	 * 
	 * 	Returns the filled option map.
	 */
	variables_map& getOptions() {
		return vm;
	};
	
	/*!
	 * 	Function: getDescription.
	 * 
	 * 	Returns the option description.
	 */
	options_description& getDescription() {
		return config;
	}
	
private:
	// Declare a group of options that will be 
        // allowed both on command line and in
        // config file
        options_description config;
	
	int opt;
	variables_map vm;
};

#endif