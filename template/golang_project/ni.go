/************************************************************************************
***
***	Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
***
***	File Author: Dell, 2020-08-03 18:09:40
***
************************************************************************************/

package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
)

var (
	help bool

	clean string
	color string

	zoom       string
	zoom_scale float64

	patch      string
	patch_mask string

	output string
)

func init() {
	flag.BoolVar(&help, "h", false, "Display this help")

	flag.StringVar(&clean, "clean", "", "Clean image")
	flag.StringVar(&color, "color", "", "Color image")

	flag.StringVar(&zoom, "zoom", "", "Zoom in/out image")
	flag.Float64Var(&zoom_scale, "scale", 4.0, "Zoom in/out scale ratio")

	flag.StringVar(&patch, "patch", "", "Patch image")
	flag.StringVar(&patch_mask, "mask", "", "Mask image")

	flag.StringVar(&output, "output", "output", "Output directory")

	// flag.PrintDefaults() is not good enough !
	flag.Usage = usage
}

func usage() {
	const version = "1.0"

	fmt.Println("New Image (ni) Version:", version)
	fmt.Println("Usage: ni [-clean image] [-color image] [-zoom image -scale ratio] [-patch image -mask maskimg] [-output dir] [-h]")
	fmt.Println("Options:")

	fmt.Println("    -h                           Display this help")
	fmt.Println("    -clean image                 Clean image")
	fmt.Println("    -color image                 Color image")
	fmt.Println("    -zoom image -scale ratio     Zoom in/out with ratio, default: 4.0")
	fmt.Println("    -patch image -mask maskimg   Patch image with mask")
	fmt.Println("    -output directory            Output directory, default: output")
}

func image_clean(image_file string, output_dir string) int {
	output_file := strings.Join([]string{output_dir, image_file}, "/")
	fmt.Printf("Clean %s, output to %s\n", image_file, output_file)
	return 0
}

func image_color(image_file string, output_dir string) int {
	output_file := strings.Join([]string{output_dir, image_file}, "/")
	fmt.Printf("Color %s, output to %s\n", image_file, output_file)
	return 0
}

func image_zoom(image_file string, scale_ratio float64, output_dir string) int {
	output_file := strings.Join([]string{output_dir, image_file}, "/")
	fmt.Printf("Zoom %s with %3.1f, output to %s\n", image_file, scale_ratio, output_file)
	return 0
}

func image_patch(image_file string, image_mask string, output_dir string) int {
	output_file := strings.Join([]string{output_dir, image_file}, "/")
	fmt.Printf("Patch %s with %s, output to %s\n", image_file, image_mask, output_file)
	return 0
}

func main() {
	flag.Parse()

	if len(clean)+len(color)+len(zoom)+len(patch) < 1 {
		usage()
		return
	}

	if len(output) > 0 {
		os.MkdirAll(output, os.ModePerm)
	}

	if len(clean) > 0 {
		image_clean(clean, output)
	}

	if len(color) > 0 {
		image_color(color, output)
	}

	if len(zoom) > 0 {
		// make sure zoom_scale >= 0.1 && zoom_scale <= 16.0
		image_zoom(zoom, zoom_scale, output)
	}

	if len(patch) > 0 {
		image_patch(patch, patch_mask, output)
	}
}
