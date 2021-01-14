/************************************************************************************
***
***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2021-01-13 21:19:01
***
************************************************************************************/


#ifndef _ENGINE_H
#define _ENGINE_H

#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_c_api.h>
// #include <cmath>
#include <vector>

#define DWORD uint32_t

// ONNX Runtime Engine
typedef struct {
	DWORD magic;
	const char *model_path;

	OrtEnv *env;
	OrtSession *session;
	OrtSessionOptions *session_options;

	 std::vector < const char *>input_node_names;
	 std::vector < int64_t > input_node_dims;	// classical model has only 1 input node {1, 3, 224, 224}.

	 std::vector < const char *>output_node_names;
	 std::vector < int64_t > output_node_dims;	// classical model has only 1 output node {1, 1000, 1, 1}.
} OrtEngine;

#define CheckEngine(e) \
    do { \
            if (! ValidEngine(e)) { \
				fprintf(stderr, "Bad OrtEngine.\n"); \
				exit(1); \
            } \
    } while(0)

void CheckStatus(OrtStatus * status);

OrtValue *CreateTensor(std::vector < int64_t > &tensor_dims, float *data, size_t size);
float *TensorValues(OrtValue * tensor);
void ReleaseTensor(OrtValue * tensor);

OrtEngine *CreateEngine(const char *model_path);
int ValidEngine(OrtEngine * engine);
OrtValue *SimpleForward(OrtEngine & engine, OrtValue * input_tensor);
void ReleaseEngine(OrtEngine * engine);

void EngineTest();

#endif							// _ENGINE_H
