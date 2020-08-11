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
int	ni_clean_init();
int	ni_sharp_init();
int	ni_color_init();
int	ni_zoom_init();
int	ni_patch_init();

// Clean
int ni_clean_file(char *infile, int sigma, char *outfile);

// Sharp
int ni_sharp_file(char *infile, int sigma, char *outfile);

// Color
int ni_color_file(char *infile, char *json, char *outfile);

// Zoom
int ni_zoom_file(char *infile, int scale, char *outfile);

// Patch
int ni_patch_file(char *infile, char *mask, char *outfile);

void ni_clean_exit();
void ni_sharp_exit();
void ni_color_exit();
void ni_zoom_exit();
void ni_patch_exit();
void image_exit();

// Funny Applications ...

// MS_END

#if defined(__cplusplus)
}
#endif

#endif	// _IMAGE_H

