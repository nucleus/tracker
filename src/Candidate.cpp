#include "Candidate.h"

Candidate::Candidate() {
	x = 0;
	y = 0;
	cantStartANewPath = false;
	pathcnt = 0;
};

Candidate::Candidate(int _x, int _y) {
	x = _x;
	y = _y;
	cantStartANewPath = false;
	pathcnt = 0;
};

