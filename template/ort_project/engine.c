/************************************************************************************
***
***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2021-01-12 23:52:44
***
************************************************************************************/

// Reference: 
// https://github.com/microsoft/onnxruntime/edit/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp

#include <assert.h>
#include <onnxruntime_c_api.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#define DWORD uint32_t

// ONNX Runtime Engine
#define MAKE_FOURCC(a,b,c,d) (((DWORD)(a) << 24) | ((DWORD)(b) << 16) | ((DWORD)(c) << 8) | ((DWORD)(d) << 0))
#define ENGINE_MAGIC MAKE_FOURCC('O', 'N', 'R', 'T')
const OrtApi *onnx_runtime_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

void CheckStatus(OrtStatus * status)
{
	if (status != NULL) {
		const char *msg = onnx_runtime_api->GetErrorMessage(status);
		fprintf(stderr, "%s\n", msg);
		onnx_runtime_api->ReleaseStatus(status);
		exit(1);
	}
}

void ReleaseTensor(OrtValue *tensor)
{
	onnx_runtime_api->ReleaseValue(tensor);
}

typedef struct {
	DWORD magic;

	OrtEnv *env;
	OrtSession *session;
	OrtSessionOptions *session_options;
	OrtAllocator *allocator;

	std::vector <const char *>input_node_names;
	std::vector < int64_t > input_node_dims;   // classical model has only 1 input node {1, 3, 224, 224}.

	std::vector <const char *>output_node_names;
} ImageCleanEngine;

void __init_input_nodes(ImageCleanEngine *t)
{
	size_t num_nodes;
	CheckStatus(onnx_runtime_api->SessionGetInputCount(t->session, &num_nodes));

	printf("Input nodes:");
	for (size_t i = 0; i < num_nodes; i++) {
		char *name;
		
		CheckStatus(onnx_runtime_api->SessionGetInputName(t->session, i, t->allocator, &name));
		t->input_node_names.push_back(name);

		OrtTypeInfo *typeinfo;
		CheckStatus(onnx_runtime_api->SessionGetInputTypeInfo(t->session, i, &typeinfo));

		const OrtTensorTypeAndShapeInfo *tensor_info;
		CheckStatus(onnx_runtime_api->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

		ONNXTensorElementDataType type;
		CheckStatus(onnx_runtime_api->GetTensorElementType(tensor_info, &type));

		size_t num_dims;
		CheckStatus(onnx_runtime_api->GetDimensionsCount(tensor_info, &num_dims));
		t->input_node_dims.resize(num_dims);

		printf("  %zu : name=%s type=%d dims=%zu: ", i, name, type, num_dims);

		CheckStatus(onnx_runtime_api->GetDimensions(tensor_info, (int64_t *) t->input_node_dims.data(), num_dims));

		for (size_t j = 0; j < num_dims; j++) {
			if (j < num_dims - 1)
				printf("%jd x ", t->input_node_dims[j]);
			else
				printf("%jd\n", t->input_node_dims[j]);
		}

		onnx_runtime_api->ReleaseTypeInfo(typeinfo);
	}
}

void __init_output_nodes(ImageCleanEngine *t)
{
	std::vector < int64_t > node_dims;

	size_t num_nodes;
	CheckStatus(onnx_runtime_api->SessionGetOutputCount(t->session, &num_nodes));

	printf("Output nodes:");
	for (size_t i = 0; i < num_nodes; i++) {
		char *name;

		CheckStatus(onnx_runtime_api->SessionGetOutputName(t->session, i, t->allocator, &name));
		t->output_node_names.push_back(name);

		OrtTypeInfo *typeinfo;
		CheckStatus(onnx_runtime_api->SessionGetInputTypeInfo(t->session, i, &typeinfo));

		const OrtTensorTypeAndShapeInfo *tensor_info;
		CheckStatus(onnx_runtime_api->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

		ONNXTensorElementDataType type;
		CheckStatus(onnx_runtime_api->GetTensorElementType(tensor_info, &type));

		size_t num_dims;
		CheckStatus(onnx_runtime_api->GetDimensionsCount(tensor_info, &num_dims));
		node_dims.resize(num_dims);

		printf("  %zu : name=%s type=%d dims=%zu: ", i, name, type, num_dims);

		CheckStatus(onnx_runtime_api->GetDimensions(tensor_info, (int64_t *) node_dims.data(), num_dims));
		for (size_t j = 0; j < num_dims; j++) {
			if (j < num_dims - 1)
				printf("%jd x ", node_dims[j]);
			else
				printf("%jd\n", node_dims[j]);
		}

		onnx_runtime_api->ReleaseTypeInfo(typeinfo);
	}
}

ImageCleanEngine *engine_create()
{
	ImageCleanEngine *t;

	t = (ImageCleanEngine *) calloc((size_t) 1, sizeof(ImageCleanEngine));
	if (! t) {
		fprintf(stderr, "Allocate memeory.");
		return NULL;
	}
	t->magic = ENGINE_MAGIC;

	// Building ...
	CheckStatus(onnx_runtime_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ImageClean", &(t->env)));

	// initialize session options if needed
	CheckStatus(onnx_runtime_api->CreateSessionOptions(&(t->session_options)));
	// onnx_runtime_api->SetIntraOpNumThreads(t->session_options, 1);

	// Sets graph optimization level
	CheckStatus(onnx_runtime_api->SetSessionGraphOptimizationLevel(t->session_options, ORT_ENABLE_BASIC));

	// Optionally add more execution providers via session_options
	// E.g. for CUDA include cuda_provider_factory.h and uncomment the following line:
	// OrtSessionOptionsAppendExecutionProvider_CUDA(t->sessionOptions, 0);

	const char *model_path = "ImageClean.onnx";
	CheckStatus(onnx_runtime_api->CreateSession(t->env, model_path, t->session_options, &(t->session)));

	CheckStatus(onnx_runtime_api->GetAllocatorWithDefaultOptions(&(t->allocator)));

	// Setup input_node_names;
	__init_input_nodes(t);

	// Setup output_node_names;
	__init_output_nodes(t);

	return t;
}

int engine_valid(ImageCleanEngine *engine)
{
	return (!engine || engine->magic != ENGINE_MAGIC) ? 0 : 1;
}

OrtValue *engine_make_input(ImageCleanEngine *t, float *data, size_t size)
{
	OrtStatus *status;
	OrtValue *input_tensor = NULL;

	OrtMemoryInfo *memory_info;
	CheckStatus(onnx_runtime_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    status = onnx_runtime_api->CreateTensorWithDataAsOrtValue(memory_info,
    	data, size * sizeof(float), 
    	t->input_node_dims.data(), 4,
    	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 
    	&input_tensor);
	CheckStatus(status);

	int is_tensor;
	CheckStatus(onnx_runtime_api->IsTensor(input_tensor, &is_tensor));
	assert(is_tensor);
	onnx_runtime_api->ReleaseMemoryInfo(memory_info);

	return input_tensor;
}

OrtValue *engine_forward(ImageCleanEngine *engine, OrtValue *input_tensor)
{
	int is_tensor;
	OrtStatus *status;
	OrtValue *output_tensor = NULL;

	CheckStatus(status = onnx_runtime_api->IsTensor(input_tensor, &is_tensor));
	assert(is_tensor);

	status = onnx_runtime_api->Run(engine->session, NULL, 
		engine->input_node_names.data(), (const OrtValue * const *) &input_tensor, 1,
		engine->output_node_names.data(), 1, &output_tensor);

	CheckStatus(status);

	CheckStatus(onnx_runtime_api->IsTensor(output_tensor, &is_tensor));
	assert(is_tensor);

	return output_tensor;
}

void engine_destroy(ImageCleanEngine *engine)
{
	if (! engine_valid(engine))
		return;

	// Release ...
	engine->input_node_names.clear();
	engine->input_node_dims.clear();
	engine->output_node_names.clear();

	onnx_runtime_api->ReleaseAllocator(engine->allocator);
	onnx_runtime_api->ReleaseSession(engine->session);
	onnx_runtime_api->ReleaseSessionOptions(engine->session_options);
	onnx_runtime_api->ReleaseEnv(engine->env);

	free(engine);
}

int main1(int argc, char *argv[])
{
	OrtEnv *env;
	CheckStatus(onnx_runtime_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

	// initialize session options if needed
	OrtSessionOptions *session_options;
	CheckStatus(onnx_runtime_api->CreateSessionOptions(&session_options));
	onnx_runtime_api->SetIntraOpNumThreads(session_options, 1);

	// Sets graph optimization level
	onnx_runtime_api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

	// Optionally add more execution providers via session_options
	// E.g. for CUDA include cuda_provider_factory.h and uncomment the following line:
	// OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

	OrtSession *session;
	const char *model_path = "squeezenet.onnx";
	CheckStatus(onnx_runtime_api->CreateSession(env, model_path, session_options, &session));

	OrtAllocator *allocator;
	CheckStatus(onnx_runtime_api->GetAllocatorWithDefaultOptions(&allocator));




	//*************************************************************************
	// print model input layer (node names, types, shape etc.)
	OrtStatus *status;

	size_t num_nodes;
	status = onnx_runtime_api->SessionGetInputCount(session, &num_nodes);
	std::vector < const char *>input_node_names(num_nodes);
	std::vector < int64_t > node_dims;	// simplify... this model has only 1 input node {1, 3, 224, 224}.
	// Otherwise need vector<vector<>>

	// iterate over all input nodes
	for (size_t i = 0; i < num_nodes; i++) {
		// print input node names
		char *name;
		status = onnx_runtime_api->SessionGetInputName(session, i, allocator, &name);
		printf("Input %zu : name=%s\n", i, name);
		input_node_names[i] = name;

		// print input node types
		OrtTypeInfo *typeinfo;
		status = onnx_runtime_api->SessionGetInputTypeInfo(session, i, &typeinfo);
		const OrtTensorTypeAndShapeInfo *tensor_info;
		CheckStatus(onnx_runtime_api->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
		ONNXTensorElementDataType type;
		CheckStatus(onnx_runtime_api->GetTensorElementType(tensor_info, &type));
		printf("Input %zu : type=%d\n", i, type);

		// print input shapes/dims
		size_t num_dims;
		CheckStatus(onnx_runtime_api->GetDimensionsCount(tensor_info, &num_dims));
		printf("Input %zu : num_dims=%zu\n", i, num_dims);
		node_dims.resize(num_dims);
		onnx_runtime_api->GetDimensions(tensor_info, (int64_t *) node_dims.data(), num_dims);
		for (size_t j = 0; j < num_dims; j++)
			printf("Input %zu : dim %zu=%jd\n", i, j, node_dims[j]);

		onnx_runtime_api->ReleaseTypeInfo(typeinfo);
	}

	// Results should be...
	// Number of inputs = 1
	// Input 0 : name = data_0
	// Input 0 : type = 1
	// Input 0 : num_dims = 4
	// Input 0 : dim 0 = 1
	// Input 0 : dim 1 = 3
	// Input 0 : dim 2 = 224
	// Input 0 : dim 3 = 224

	//*************************************************************************
	// Similar operations to get output node information.
	// Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
	// OrtSessionGetOutputTypeInfo() as shown above.

	//*************************************************************************
	// Score the model using sample data, and inspect values

	// OrtValue *CreateTensor()

	size_t input_tensor_size = 224 * 224 * 3;	// simplify ... using known dim values to calculate size
	// use OrtGetTensorShapeElementCount() to get official size!
	std::vector < float >input_tensor_values(input_tensor_size);
	// initialize input data with values in [0.0, 1.0]
	for (size_t i = 0; i < input_tensor_size; i++)
		input_tensor_values[i] = (float) i / (input_tensor_size + 1);

	// create input tensor object from data values
	OrtMemoryInfo *memory_info;
	CheckStatus(onnx_runtime_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
	OrtValue *input_tensor = NULL;
	CheckStatus(onnx_runtime_api->
				CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(),
											   input_tensor_size * sizeof(float), node_dims.data(), 4,
											   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
	int is_tensor;
	CheckStatus(onnx_runtime_api->IsTensor(input_tensor, &is_tensor));
	assert(is_tensor);
	onnx_runtime_api->ReleaseMemoryInfo(memory_info);



	// score model & input tensor, get back output tensor
	std::vector < const char *>output_node_names = { "softmaxout_1" };
	OrtValue *output_tensor = NULL;
	CheckStatus(onnx_runtime_api->
				Run(session, NULL, input_node_names.data(), (const OrtValue * const *) &input_tensor, 1,
					output_node_names.data(), 1, &output_tensor));
	CheckStatus(onnx_runtime_api->IsTensor(output_tensor, &is_tensor));
	assert(is_tensor);

	// Get pointer to output tensor float values
	float *floatarr;
	CheckStatus(onnx_runtime_api->GetTensorMutableData(output_tensor, (void **) &floatarr));
	assert(std::abs(floatarr[0] - 0.000045) < 1e-6);

	// score the model, and print scores for first 5 classes
	for (int i = 0; i < 5; i++)
		printf("Score for class [%d] =  %f\n", i, floatarr[i]);

	// Results should be as below...
	// Score for class[0] = 0.000045
	// Score for class[1] = 0.003846
	// Score for class[2] = 0.000125
	// Score for class[3] = 0.001180
	// Score for class[4] = 0.001317

	onnx_runtime_api->ReleaseValue(output_tensor);
	onnx_runtime_api->ReleaseValue(input_tensor);


	onnx_runtime_api->ReleaseSession(session);
	onnx_runtime_api->ReleaseSessionOptions(session_options);
	onnx_runtime_api->ReleaseEnv(env);
	printf("Done!\n");
	return 0;
}
