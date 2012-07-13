/*
 *	File: global.h
 *	---------------------------
 *	This file contains global variable definitions, preprocessor defines and macros.
 *
 *	Author: Michael Andersch, 2012
 */
#ifndef GLOBAL_H_
#define GLOBAL_H_

#define CONFIG_FILE ("options.cfg")

#define DEFAULT_CAMERA (0)

#define ERROR(e) (string("ERROR: ") + string(e))
#define abs(x) ((x) < 0 ? -(x) : (x))

#define DEFAULT_MODEL_TRAINING_COUNT (100)
#define GAUSSIAN_WINDOW_RADIUS (5)
#define GAUSSIAN_WIDTH_CAND (15)
#define THRESH (128.0f)
#define DEFAULT_LEARNING_RATE (0.02)
#define ALLOWED_CANDIDATES (1024)

#define PACKU32(x,y) ((x << 16) | (y & 0xFFFF))
#define UNPACKHI32(x) (x >> 16)
#define UNPACKLO32(x) (x & 0xFFFF)

#define FS_BLOCKSIZE_X 16
#define FS_BLOCKSIZE_Y 16
#define RGB2GRAY_BS_X 128
#define RGB2GRAY_BS_Y 1

#define INITIAL_FPS 100

#define PIPELINE_BUFFER_SIZE 10

#endif