/************************************************************************************
***
***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2020-08-11 19:54:36
***
************************************************************************************/

#include "image.h"

// One-stop header.
#include <torch/script.h>

// headers for opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define IMAGE_SIZE 224
#define CHANNELS 3

int load_image(std::string file_name, cv::Mat &image)
{
	image = cv::imread(file_name);	// CV_8UC3

	if (image.empty() || !image.data) {
		std::cerr << "Loading image " << file_name << " error." << std::endl;
		return IMAGE_ERROR;
	}
	cv::cvtColor(image, image, CV_BGR2RGB);

	// scale image to fit
	// cv::Size scale(IMAGE_SIZE, IMAGE_SIZE);
	// cv::resize(image, image, scale);

	// convert [unsigned int] to [float]
	image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

	return IMAGE_OK;
}

int load_model(std::string filename, torch::jit::script::Module &module)
{
	try {
		module = torch::jit::load(filename);
	}
	catch(const c10::Error & e) {
		std::cerr << "Loading model " << filename << " error." << std::endl;
		exit(-1);	// Fatal error.
		return IMAGE_ERROR;
	}

	// to GPU
	module.to(at::kCUDA);

	return IMAGE_OK;
}

void release_model(torch::jit::script::Module &module)
{
	// Delete module;
	// del module;
}

// Clean
static int clean_flag = 0;
static torch::jit::script::Module clean_model;

int ni_clean_init()
{
	if (! clean_flag) {
		load_model("clean.pt", clean_model);
		clean_flag = 1;
	}
	return IMAGE_OK;
}

int ni_clean_file(char *infile, int sigma, char *outfile)
{
	cv::Mat image;

	printf("Cleaning %s with %d to %s ...\n", infile, sigma, outfile);

	if (load_image(infile, image) != 0)
		return IMAGE_ERROR;

	auto input_tensor = torch::from_blob(image.data, { 1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS });
	input_tensor = input_tensor.permute( {0, 3, 1, 2} );
	// input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
	// input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
	// input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

	// to GPU
	input_tensor = input_tensor.to(at::kCUDA);

	torch::Tensor output_tensor;
	output_tensor = clean_model.forward( {input_tensor} ).toTensor();

	std::cout << "Output Result:" << std::endl;

	return IMAGE_OK;
}

void ni_clean_exit()
{
	if (clean_flag) {
		release_model(clean_model);
	}
	clean_flag = 0;
}

// Sharp
static int sharp_flag = 0;
static torch::jit::script::Module sharp_model;

int ni_sharp_init()
{
	if (! sharp_flag) {
		load_model("sharp.pt", sharp_model);
		sharp_flag = 1;
	}
	return IMAGE_OK;
}

int ni_sharp_file(char *infile, int sigma, char *outfile)
{
	printf("Sharping %s with %d to %s ...\n", infile, sigma, outfile);
	return IMAGE_OK;
}

void ni_sharp_exit()
{
	if (sharp_flag) {
		release_model(sharp_model);
	}
	sharp_flag = 0;
}

// Color
static int color_flag = 0;
static torch::jit::script::Module color_model;

int ni_color_init()
{
	if (! color_flag) {
		load_model("color.pt", sharp_model);
		color_flag = 1;
	}
	return IMAGE_OK;
}

int ni_color_file(char *infile, char *json, char *outfile)
{
	printf("Coloring %s with %s to %s ...\n", infile, json, outfile);
	return IMAGE_OK;
}

void ni_color_exit()
{
	if (color_flag) {
		release_model(color_model);
	}
	color_flag = 0;
}

// Zoom
static int zoom_flag = 0;
static torch::jit::script::Module zoom_model;

int ni_zoom_init()
{
	if (! zoom_flag) {
		load_model("zoom.pt", sharp_model);
		zoom_flag = 1;
	}
	return IMAGE_OK;
}

int ni_zoom_file(char *infile, int scale, char *outfile)
{
	printf("Zooming %s with %d to %s ...\n", infile, scale, outfile);
	return IMAGE_OK;
}

void ni_zoom_exit()
{
	if (zoom_flag) {
		release_model(zoom_model);
	}
	zoom_flag = 0;
}

// Patch
static int patch_flag = 0;
static torch::jit::script::Module patch_model;

int ni_patch_init()
{
	if (! patch_flag) {
		load_model("patch.pt", patch_model);
		patch_flag = 1;
	}
	return IMAGE_OK;
}

int ni_patch_file(char *infile, char *mask, char *outfile)
{
	printf("Patching %s with %s to %s ...\n", infile, mask, outfile);
	return IMAGE_OK;
}

void ni_patch_exit()
{
	if (patch_flag) {
		release_model(patch_model);
	}
	patch_flag = 0;
}

// Interface
void image_init()
{
	ni_clean_init();
	ni_sharp_init();
	ni_color_init();
	ni_zoom_init();
	ni_patch_init();
}

void image_exit()
{
	ni_clean_exit();
	ni_sharp_exit();
	ni_color_exit();
	ni_zoom_exit();
	ni_patch_exit();
}
