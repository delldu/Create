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

	zoom  string
	slow  string
	scale float64

	output string
)

func init() {
	flag.BoolVar(&help, "h", false, "Display this help")

	flag.StringVar(&clean, "clean", "", "Clean video")
	flag.StringVar(&color, "color", "", "Color video")

	flag.StringVar(&zoom, "zoom", "", "Zoom in/out video")
	flag.StringVar(&slow, "slow", "", "Slow video")
	flag.Float64Var(&scale, "scale", 4.0, "Zoom in/out or slow scale ratio")

	flag.StringVar(&output, "output", "output", "Output directory")

	// flag.PrintDefaults() is not good enough !
	flag.Usage = usage
}

func usage() {
	const version = "1.0"

	fmt.Println("New Video (nv) Version:", version)
	fmt.Println("Usage: nv [-clean video] [-color video] [-zoom video -scale ratio] [-slow video -scale ratio] [-output dir] [-h]")
	fmt.Println("Options:")

	fmt.Println("    -h                           Display this help")
	fmt.Println("    -clean video                 Clean video")
	fmt.Println("    -color video                 Color video")
	fmt.Println("    -zoom video -scale ratio     Zoom in/out with scale, default: 4.0")
	fmt.Println("    -slow video -scale ratio     Slow video with ratio, default: 4.0")
	fmt.Println("    -output directory            Output directory, default: output")
}

func video_clean(video_file string, output_dir string) int {
	output_file := strings.Join([]string{output_dir, video_file}, "/")
	fmt.Printf("Clean %s, output to %s\n", video_file, output_file)
	return 0
}

func video_color(video_file string, output_dir string) int {
	output_file := strings.Join([]string{output_dir, video_file}, "/")
	fmt.Printf("Color %s, output to %s\n", video_file, output_file)
	return 0
}

func video_slow(video_file string, scale_ratio float64, output_dir string) int {
	output_file := strings.Join([]string{output_dir, video_file}, "/")
	fmt.Printf("Slow %s with %3.1f, output to %s\n", video_file, scale_ratio, output_file)
	return 0
}

func video_zoom(video_file string, scale_ratio float64, output_dir string) int {
	output_file := strings.Join([]string{output_dir, video_file}, "/")
	fmt.Printf("Zoom %s with %3.1f, output to %s\n", video_file, scale_ratio, output_file)
	return 0
}

func main() {
	flag.Parse()

	if len(clean)+len(color)+len(zoom)+len(slow) < 1 {
		usage()
		return
	}

	if len(output) > 0 {
		os.MkdirAll(output, os.ModePerm)
	}

	if len(clean) > 0 {
		video_clean(clean, output)
	}

	if len(color) > 0 {
		video_color(color, output)
	}

	if len(zoom) > 0 {
		// make sure scale >= 0.1 && zoom_scale <= 16.0
		video_zoom(zoom, scale, output)
	}

	if len(slow) > 0 {
		// make sure scale >= 0.1 && zoom_scale <= 16.0
		video_slow(slow, scale, output)
	}
}
