/************************************************************************************
***
***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2020-08-11 19:56:16
***
************************************************************************************/


#ifndef _VIDEO_H
#define _VIDEO_H

#if defined(__cplusplus)
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

#define VIDEO_OK 0
#define VIDEO_ERROR (-1)
#define VIDEO_VERSION "1.0.0"

// MS_BEGIN
char *video_version();

// MS_END

#if defined(__cplusplus)
}
#endif

#endif	// _VIDEO_H

