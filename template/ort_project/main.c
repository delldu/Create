/************************************************************************************
***
***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2021-01-12 23:52:44
***
************************************************************************************/


#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>

#include "engine.h"
#include <nimage/image.h>

OrtValue *ImageToTensor(IMAGE *image)
{
	int i, j;
	float *data, *d;
	OrtValue *tensor;

	CHECK_IMAGE(image);

	std::vector<int64_t> dims;
	dims.push_back(1);		// batch
	dims.push_back(4);		// channels -- RGBA
	dims.push_back(image->height);
	dims.push_back(image->width);

	data = (float *)malloc(1 * 4 * image->height * image->width * sizeof(float));
	if (data == NULL) {
		fprintf(stderr, "Allocate memory error.\n");
		exit(-1);
	}

	// R
	d = data;
	image_foreach(image, i, j)
		*d++ = (float)(image->ie[i][j].r)/255.0;
	// G
	d = data + 1 * image->height * image->width;
	image_foreach(image, i, j)
		*d++ = (float)(image->ie[i][j].g)/255.0;
	// B
	d = data + 2 * image->height * image->width;
	image_foreach(image, i, j)
		*d++ = (float)(image->ie[i][j].b)/255.0;
	// A
	d = data + 3 * image->height * image->width;
	image_foreach(image, i, j)
		*d++ = (float)(image->ie[i][j].a)/255.0;

	tensor = CreateTensor(dims, data, 1 * 4 * image->height * image->width);

	// free(data); xxxx8888

	return tensor;
}

IMAGE *TensorToImage(OrtValue *tensor)
{
	int i, j;
	IMAGE *image;
	float *data, *d;

	CheckTensor(tensor);
	std::vector<int64_t> dims = TensorDimensions(tensor);
	data = TensorValues(tensor);

	image = image_create((int16_t)dims[2], (int16_t)dims[3]); CHECK_IMAGE(image);
	// R
	d = data;
	image_foreach(image, i, j) {
		image->ie[i][j].r = (uint8_t)(*d * 255.0); d++;
	}
	// G
	d = data + 1 * dims[2] * dims[3];
	image_foreach(image, i, j) {
		image->ie[i][j].g = (uint8_t)(*d * 255.0); d++;
	}
	// B
	d = data + 2 * dims[2] * dims[3];
	image_foreach(image, i, j) {
		image->ie[i][j].b = (uint8_t)(*d * 255.0); d++;
	}
	// A ?

	if (dims[1] > 3) {
		// A
		d = data + 3 * dims[2] * dims[3];
		image_foreach(image, i, j) {
			image->ie[i][j].a = (uint8_t)(*d * 255.0); d++;
		}
	} else {
		image_foreach(image, i, j)
			image->ie[i][j].a = 255;
	}

	return image;
}


void help(char *cmd)
{
	printf("Usage: %s [option]\n", cmd);
	printf("    -h, --help                   Display this help.\n");
	printf("    -m, --model <model.onnx>     Model name of onnx.\n");
	printf("    -i, --input <png>            Input image.\n");
	printf("    -o, --output <file>          Output image.\n");

	exit(1);
}

int main(int argc, char **argv)
{
	int optc;
	int option_index = 0;
	char *model_onnx = "demo.onnx";
	char *input_png = NULL;
	char *output_file = "output.png";

	struct option long_opts[] = {
		{"help", 0, 0, 'h'},
		{"model", 1, 0, 'm'},
		{"input", 1, 0, 'i'},
		{"output", 1, 0, 'o'},
		{0, 0, 0, 0}
	};

	while ((optc = getopt_long(argc, argv, "h m: i: o:", long_opts, &option_index)) != EOF) {
		switch (optc) {
		case 'm':
			model_onnx = optarg;
			break;
		case 'i':
			input_png = optarg;
			break;
		case 'o':
			output_file = optarg;
			break;
		case 'h':				// help
		default:
			help(argv[0]);
			break;
		}
	}

	if (! input_png) {
		help(argv[0]);
		return -1;
	}

	// EngineTest();

	OrtEngine *engine = CreateEngine(model_onnx); CheckEngine(engine);

	IMAGE *image = image_load(input_png); check_image(image);
	OrtValue *input_tensor = ImageToTensor(image);
	image_destroy(image);

	OrtValue *output_tensor;

	// Test speed ...
	int k;
	for (k = 0; k < 10; k++) {
		printf("%d ...\n", k);
		output_tensor = SimpleForward(engine, input_tensor);
		ReleaseTensor(output_tensor);
	}

	output_tensor = SimpleForward(engine, input_tensor);

	IMAGE *output_image = TensorToImage(output_tensor); check_image(output_image);
	image_save(output_image, output_file);
	image_destroy(output_image);
	ReleaseTensor(output_tensor);

	ReleaseTensor(input_tensor);

	ReleaseEngine(engine);


	return 0;
}
