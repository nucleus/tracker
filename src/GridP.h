#ifndef GRIDP_H
#define GRIDP_H

#define GRID_CELLSIZE 16

#include "Path.h"

#include <Candidate.h>

#include <list>

using namespace std;

/*!
 * 	Class: GridP.
 * 
 * 	This class represents the grid data structure for paths.
 */
class GridP {
	public:
		GridP(void);
		GridP(int width, int height);
		
		/*!
		 * 	Function: GetEntry.
		 * 
		 * 	Returns the grid entry corresponding to the given (x,y) location. 
		 */
		list<Path> * GetEntry(int x, int y);
		
		/*!
		 * 	Function: SetEntry.
		 * 
		 * 	Adds an entry to the grid cell with the given (x,y) location.
		 */
		void SetEntry(int x, int y, list<Path> * entry);
		int xcells;
		int ycells;
	private:
		list<Path> * * grid;
};

#endif

