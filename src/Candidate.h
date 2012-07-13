#ifndef CANDIDATE_H
#define CANDIDATE_H

/*!
 * 	Class: Candidate.
 * 
 * 	Represents a ball candidate. Each candidate is identified by its (x,y) location.
 */
class Candidate {
	public:
		Candidate();
		Candidate(int _x, int _y);
		int x;
		int y;
		bool cantStartANewPath;
		int pathcnt;
		
};

#endif
