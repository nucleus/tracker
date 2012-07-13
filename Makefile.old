CFLAGS  = -std=c++0x -O3 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video
LDFLAGS = -O3 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video
SOURCES = ForegroundSegmenter.cpp VideoBackend.cpp main.cpp BallDetection.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = tracker

all: $(SOURCES) $(EXECUTABLE)
 
$(EXECUTABLE): $(OBJECTS)  
	g++ $(LDFLAGS) $(OBJECTS) -o $@

%.o: %.cpp
	g++ $(CFLAGS) -c $< -o $@ 

clean:
	rm -rf *.o $(EXECUTABLE)

