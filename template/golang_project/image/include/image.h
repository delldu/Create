/************************************************************************************
***
***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2020-08-11 19:53:54
***
************************************************************************************/


#ifndef _IMAGE_H
#define _IMAGE_H

#if defined(__cplusplus)
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

#define IMAGE_OK 0
#define IMAGE_ERROR (-1)
#define IMAGE_VERSION "1.0.0"

// MS_BEGIN
void image_init();

// Clean
int clean_file(char *infile, int sigma, char *outfile);

// Sharp
int sharp_file(char *infile, int sigma, char *outfile);

// Color
int color_file(char *infile, char *json, char *outfile);

// Zoom
int zoom_file(char *infile, int scale, char *outfile);

// Patch
int patch_file(char *infile, char *mask, char *outfile);

void image_exit();


// Funny Applications ...

// MS_END

#if defined(__cplusplus)
}
#endif

#endif	// _IMAGE_H

