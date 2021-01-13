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
#include "engine.h"

// ONNX Runtime Engine
#define MAKE_FOURCC(a,b,c,d) (((DWORD)(a) << 24) | ((DWORD)(b) << 16) | ((DWORD)(c) << 8) | ((DWORD)(d) << 0))
#define ENGINE_MAGIC MAKE_FOURCC('O', 'N', 'R', 'T')
const OrtApi *onnx_runtime_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

void SetInputNodes(OrtEngine *t)
{
	size_t num_nodes;
	OrtAllocator *allocator;
	CheckStatus(onnx_runtime_api->GetAllocatorWithDefaultOptions(&allocator));

	CheckStatus(onnx_runtime_api->SessionGetInputCount(t->session, &num_nodes));

	printf("Input nodes:\n");
	for (size_t i = 0; i < num_nodes; i++) {
		char *name;
		
		CheckStatus(onnx_runtime_api->SessionGetInputName(t->session, i, allocator, &name));
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

		printf("    no=%zu name=\"%s\" type=%d dims=%zu: ", i, name, type, num_dims);

		CheckStatus(onnx_runtime_api->GetDimensions(tensor_info, (int64_t *) t->input_node_dims.data(), num_dims));

		for (size_t j = 0; j < num_dims; j++) {
			if (j < num_dims - 1)
				printf("%jd x ", t->input_node_dims[j]);
			else
				printf("%jd\n", t->input_node_dims[j]);
		}

		onnx_runtime_api->ReleaseTypeInfo(typeinfo);
	}
	// onnx_runtime_api->ReleaseAllocator(allocator); segmant fault !!!
}

void SetOutputNodes(OrtEngine *t)
{
	OrtAllocator *allocator;

	CheckStatus(onnx_runtime_api->GetAllocatorWithDefaultOptions(&allocator));

	size_t num_nodes;
	CheckStatus(onnx_runtime_api->SessionGetOutputCount(t->session, &num_nodes));

	printf("Output nodes:\n");
	for (size_t i = 0; i < num_nodes; i++) {
		char *name;

		CheckStatus(onnx_runtime_api->SessionGetOutputName(t->session, i, allocator, &name));
		t->output_node_names.push_back(name);

		OrtTypeInfo *typeinfo;
		CheckStatus(onnx_runtime_api->SessionGetOutputTypeInfo(t->session, i, &typeinfo));

		const OrtTensorTypeAndShapeInfo *tensor_info;
		CheckStatus(onnx_runtime_api->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

		ONNXTensorElementDataType type;
		CheckStatus(onnx_runtime_api->GetTensorElementType(tensor_info, &type));

		size_t num_dims;
		CheckStatus(onnx_runtime_api->GetDimensionsCount(tensor_info, &num_dims));
		t->output_node_dims.resize(num_dims);

		printf("    no=%zu name=\"%s\" type=%d dims=%zu: ", i, name, type, num_dims);

		CheckStatus(onnx_runtime_api->GetDimensions(tensor_info, (int64_t *) t->output_node_dims.data(), num_dims));
		for (size_t j = 0; j < num_dims; j++) {
			if (j < num_dims - 1)
				printf("%jd x ", t->output_node_dims[j]);
			else
				printf("%jd\n", t->output_node_dims[j]);
		}

		onnx_runtime_api->ReleaseTypeInfo(typeinfo);
	}
	// onnx_runtime_api->ReleaseAllocator(allocator); segmant fault !!!
}

void CheckStatus(OrtStatus * status)
{
	if (status != NULL) {
		const char *msg = onnx_runtime_api->GetErrorMessage(status);
		fprintf(stderr, "%s\n", msg);
		onnx_runtime_api->ReleaseStatus(status);
		exit(1);
	}
}

OrtValue *CreateTensor(std::vector<int64_t> &tensor_dims, float *data, size_t size)
{
	OrtStatus *status;
	OrtValue *tensor = NULL;

	OrtMemoryInfo *memory_info;
	CheckStatus(onnx_runtime_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    status = onnx_runtime_api->CreateTensorWithDataAsOrtValue(memory_info,
    	data, size * sizeof(float), 
    	tensor_dims.data(), tensor_dims.size(),
    	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 
    	&tensor);
	CheckStatus(status);
	onnx_runtime_api->ReleaseMemoryInfo(memory_info);

	int is_tensor;
	CheckStatus(onnx_runtime_api->IsTensor(tensor, &is_tensor));
	assert(is_tensor);

	return tensor;
}

float *TensorValues(OrtValue *tensor)
{
	float *floatarray;
	CheckStatus(onnx_runtime_api->GetTensorMutableData(tensor, (void **) &floatarray));
	return floatarray;
}

void ReleaseTensor(OrtValue *tensor)
{
	onnx_runtime_api->ReleaseValue(tensor);
}

OrtEngine *CreateEngine(const char *model_path)
{
	OrtEngine *t;

	printf("Creating ONNX Runtime Engine for model %s ...\n", model_path);

	t = (OrtEngine *) calloc((size_t) 1, sizeof(OrtEngine));
	if (! t) {
		fprintf(stderr, "Allocate memeory.");
		return NULL;
	}
	t->magic = ENGINE_MAGIC;
	t->model_path = model_path;

	// Building ...
	CheckStatus(onnx_runtime_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "OrtEngine", &(t->env)));

	// initialize session options if needed
	CheckStatus(onnx_runtime_api->CreateSessionOptions(&(t->session_options)));
	// CheckStatus(onnx_runtime_api->SetIntraOpNumThreads(t->session_options, 0));	// 0 -- for default 

	// Sets graph optimization level
	CheckStatus(onnx_runtime_api->SetSessionGraphOptimizationLevel(t->session_options, ORT_ENABLE_BASIC));
	// ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL

	// Optionally add more execution providers via session_options
	// E.g. for CUDA include cuda_provider_factory.h and uncomment the following line:
	// OrtSessionOptionsAppendExecutionProvider_CUDA(t->sessionOptions, 0);

	CheckStatus(onnx_runtime_api->CreateSession(t->env, model_path, t->session_options, &(t->session)));

	// Setup input_node_names;
	SetInputNodes(t);

	// Setup output_node_names;
	SetOutputNodes(t);

	printf("Create OK.\n");

	return t;
}

int ValidEngine(OrtEngine *t)
{
	return (!t || t->magic != ENGINE_MAGIC) ? 0 : 1;
}

// SimpleForward
OrtValue *SimpleForward(OrtEngine *engine, OrtValue *input_tensor)
{
	int is_tensor;
	OrtStatus *status;
	OrtValue *output_tensor = NULL;

	CheckStatus(status = onnx_runtime_api->IsTensor(input_tensor, &is_tensor));
	assert(is_tensor);

	/* prototype
	ORT_API2_STATUS(Run, _Inout_ OrtSession* sess, _In_opt_ const OrtRunOptions* run_options,
      _In_reads_(input_len) const char* const* input_names,
      _In_reads_(input_len) const OrtValue* const* input, size_t input_len,
      _In_reads_(output_names_len) const char* const* output_names1, size_t output_names_len,
      _Inout_updates_all_(output_names_len) OrtValue** output);
	*/
	status = onnx_runtime_api->Run(engine->session, NULL, 
		engine->input_node_names.data(), (const OrtValue * const *) &input_tensor, 1,
		engine->output_node_names.data(), 1, &output_tensor);

	CheckStatus(status);

	CheckStatus(onnx_runtime_api->IsTensor(output_tensor, &is_tensor));
	assert(is_tensor);

	return output_tensor;
}

void ReleaseEngine(OrtEngine *engine)
{
	if (! ValidEngine(engine))
		return;

	// Release ...
	engine->input_node_names.clear();
	engine->input_node_dims.clear();
	engine->output_node_names.clear();
	engine->output_node_dims.clear();

	onnx_runtime_api->ReleaseSession(engine->session);
	onnx_runtime_api->ReleaseSessionOptions(engine->session_options);
	onnx_runtime_api->ReleaseEnv(engine->env);

	free(engine);
}

void EngineTest()
{
	OrtEngine *engine;

	engine = CreateEngine("squeezenet.onnx");
	CheckEngine(engine);

	size_t input_tensor_size = 224 * 224 * 3;	// simplify ... using known dim values to calculate size
	// use OrtGetTensorShapeElementCount() to get official size!
	std::vector < float >input_tensor_values(input_tensor_size);
	// initialize input data with values in [0.0, 1.0]
	for (size_t i = 0; i < input_tensor_size; i++)
		input_tensor_values[i] = (float) i / (input_tensor_size + 1);

	OrtValue *input_tensor = CreateTensor(engine->input_node_dims, input_tensor_values.data(), input_tensor_size);

	OrtValue *output_tensor = SimpleForward(engine, input_tensor);

	float *f = TensorValues(output_tensor);

	for (int i = 0; i < 5; i++) {
		printf("Score for class [%d] =  %f\n", i, f[i]);
	}

	ReleaseTensor(input_tensor);
	ReleaseTensor(output_tensor);

	ReleaseEngine(engine);	
}
