#ifndef GRIDC_H
#define GRIDC_H

#define GRID_CELLSIZE 16

#include <Candidate.h>

#include <list>

using namespace std;

/*!
 * 	Class: GridC.
 * 
 * 	This class represents the grid data structure for candidates.
 */
class GridC {
	public:
		GridC(void);
		GridC(int width, int height);
		
		/*!
		 * 	Function: GetEntry.
		 * 
		 * 	Returns the grid entry corresponding to the given (x,y) location. 
		 */
		list<Candidate> * GetEntry(int x, int y);
		
		/*!
		 * 	Function: SetEntry.
		 * 
		 * 	Adds an entry to the grid cell with the given (x,y) location.
		 */
		void SetEntry(int x, int y, list<Candidate> * entry);
		int xcells;
		int ycells;
	private:
		list<Candidate> * * grid;
};

#endif

