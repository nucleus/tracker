# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canoncical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/michael/Documents/tracker

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/michael/Documents/tracker

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/cmake-gui -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/michael/Documents/tracker/CMakeFiles /home/michael/Documents/tracker/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/michael/Documents/tracker/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named tracker

# Build rule for target.
tracker: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 tracker
.PHONY : tracker

# fast build rule for target.
tracker/fast:
	$(MAKE) -f CMakeFiles/tracker.dir/build.make CMakeFiles/tracker.dir/build
.PHONY : tracker/fast

src/BallDetection.o: src/BallDetection.cpp.o
.PHONY : src/BallDetection.o

# target to build an object file
src/BallDetection.cpp.o:
	$(MAKE) -f CMakeFiles/tracker.dir/build.make CMakeFiles/tracker.dir/src/BallDetection.cpp.o
.PHONY : src/BallDetection.cpp.o

src/BallDetection.i: src/BallDetection.cpp.i
.PHONY : src/BallDetection.i

# target to preprocess a source file
src/BallDetection.cpp.i:
	$(MAKE) -f CMakeFiles/tracker.dir/build.make CMakeFiles/tracker.dir/src/BallDetection.cpp.i
.PHONY : src/BallDetection.cpp.i

src/BallDetection.s: src/BallDetection.cpp.s
.PHONY : src/BallDetection.s

# target to generate assembly for a file
src/BallDetection.cpp.s:
	$(MAKE) -f CMakeFiles/tracker.dir/build.make CMakeFiles/tracker.dir/src/BallDetection.cpp.s
.PHONY : src/BallDetection.cpp.s

src/ForegroundSegmenter.o: src/ForegroundSegmenter.cpp.o
.PHONY : src/ForegroundSegmenter.o

# target to build an object file
src/ForegroundSegmenter.cpp.o:
	$(MAKE) -f CMakeFiles/tracker.dir/build.make CMakeFiles/tracker.dir/src/ForegroundSegmenter.cpp.o
.PHONY : src/ForegroundSegmenter.cpp.o

src/ForegroundSegmenter.i: src/ForegroundSegmenter.cpp.i
.PHONY : src/ForegroundSegmenter.i

# target to preprocess a source file
src/ForegroundSegmenter.cpp.i:
	$(MAKE) -f CMakeFiles/tracker.dir/build.make CMakeFiles/tracker.dir/src/ForegroundSegmenter.cpp.i
.PHONY : src/ForegroundSegmenter.cpp.i

src/ForegroundSegmenter.s: src/ForegroundSegmenter.cpp.s
.PHONY : src/ForegroundSegmenter.s

# target to generate assembly for a file
src/ForegroundSegmenter.cpp.s:
	$(MAKE) -f CMakeFiles/tracker.dir/build.make CMakeFiles/tracker.dir/src/ForegroundSegmenter.cpp.s
.PHONY : src/ForegroundSegmenter.cpp.s

src/VideoBackend.o: src/VideoBackend.cpp.o
.PHONY : src/VideoBackend.o

# target to build an object file
src/VideoBackend.cpp.o:
	$(MAKE) -f CMakeFiles/tracker.dir/build.make CMakeFiles/tracker.dir/src/VideoBackend.cpp.o
.PHONY : src/VideoBackend.cpp.o

src/VideoBackend.i: src/VideoBackend.cpp.i
.PHONY : src/VideoBackend.i

# target to preprocess a source file
src/VideoBackend.cpp.i:
	$(MAKE) -f CMakeFiles/tracker.dir/build.make CMakeFiles/tracker.dir/src/VideoBackend.cpp.i
.PHONY : src/VideoBackend.cpp.i

src/VideoBackend.s: src/VideoBackend.cpp.s
.PHONY : src/VideoBackend.s

# target to generate assembly for a file
src/VideoBackend.cpp.s:
	$(MAKE) -f CMakeFiles/tracker.dir/build.make CMakeFiles/tracker.dir/src/VideoBackend.cpp.s
.PHONY : src/VideoBackend.cpp.s

src/main.o: src/main.cpp.o
.PHONY : src/main.o

# target to build an object file
src/main.cpp.o:
	$(MAKE) -f CMakeFiles/tracker.dir/build.make CMakeFiles/tracker.dir/src/main.cpp.o
.PHONY : src/main.cpp.o

src/main.i: src/main.cpp.i
.PHONY : src/main.i

# target to preprocess a source file
src/main.cpp.i:
	$(MAKE) -f CMakeFiles/tracker.dir/build.make CMakeFiles/tracker.dir/src/main.cpp.i
.PHONY : src/main.cpp.i

src/main.s: src/main.cpp.s
.PHONY : src/main.s

# target to generate assembly for a file
src/main.cpp.s:
	$(MAKE) -f CMakeFiles/tracker.dir/build.make CMakeFiles/tracker.dir/src/main.cpp.s
.PHONY : src/main.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... tracker"
	@echo "... src/BallDetection.o"
	@echo "... src/BallDetection.i"
	@echo "... src/BallDetection.s"
	@echo "... src/ForegroundSegmenter.o"
	@echo "... src/ForegroundSegmenter.i"
	@echo "... src/ForegroundSegmenter.s"
	@echo "... src/VideoBackend.o"
	@echo "... src/VideoBackend.i"
	@echo "... src/VideoBackend.s"
	@echo "... src/main.o"
	@echo "... src/main.i"
	@echo "... src/main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

