/************************************************************************************
***
***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2021-01-13 21:19:01
***
************************************************************************************/


#ifndef _ENGINE_H
#define _ENGINE_H

// #if defined(__cplusplus)
// extern "C" {
// #endif

#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_c_api.h>
// #include <cmath>
#include <vector>

#define DWORD uint32_t

// ONNX Runtime Engine
typedef struct {
	DWORD magic;

	OrtEnv *env;
	OrtSession *session;
	OrtSessionOptions *session_options;

	std::vector <const char *>input_node_names;
	std::vector < int64_t > input_node_dims;   // classical model has only 1 input node {1, 3, 224, 224}.

	std::vector <const char *>output_node_names;
	std::vector < int64_t > output_node_dims;   // classical model has only 1 output node {1, 1000, 1, 1}.
} ImageCleanEngine;

#define CheckEngine(e) \
    do { \
            if (! engine_valid(e)) { \
				fprintf(stderr, "Bad ImageCleanEngine.\n"); \
				exit(1); \
            } \
    } while(0)



void CheckStatus(OrtStatus * status);

// TensorValues ?
// CreateTensor(tensor_dims, float *data, size_t size)
OrtValue *engine_make_input(ImageCleanEngine *t, float *data, size_t size);
float *FloatValues(OrtValue *tensor);
void ReleaseTensor(OrtValue *tensor);

int engine_valid(ImageCleanEngine *engine);
// ValidEngine()



OrtValue *engine_forward(ImageCleanEngine *engine, OrtValue *input_tensor);
// EngineForward()

ImageCleanEngine *engine_create();
// CreateEngine()

void engine_destroy(ImageCleanEngine *engine);
// ReleaseEngine()


void test();

// #if defined(__cplusplus)
// }
// #endif

#endif	// _ENGINE_H
