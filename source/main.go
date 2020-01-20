/************************************************************************************
***
***	Copyright 2019 Dell(18588220928@163.com), All Rights Reserved.
***
***	File Author: Dell, 2019-08-08 23:02:07
***
************************************************************************************/

package main

import (
	"bytes"
	"bufio"
	"flag"
	"fmt"
	// "io/ioutil"
	"log"
	"strings"
	"os"
	"os/exec"
	"path/filepath"
	"text/template"
)

var (
	help  bool
	fname string
	value string
)

func init() {
	flag.BoolVar(&help, "h", false, "Display this help")
	flag.StringVar(&fname, "f", "", "Template file name")
	flag.StringVar(&value, "v", "", "Current value")

	flag.Usage = usage
}

func usage() {
	const version = "1.0"

	fmt.Println("create_temp version:", version)
	fmt.Println("Usage: create_temp [-f file] [-v value] [-h]")
	fmt.Println("Options:")

	// flag.PrintDefaults() format is not good enough !
	fmt.Println("    -h                   Display this help")
	fmt.Println("    -f file              Template file name")
	fmt.Println("    -v value             Current value")
}

func checkerror(err error) {
	if err != nil {
		log.Fatalln(err)
	}
}

func input(prompt string) string {
	if len(prompt) < 1 {
		fmt.Fprintf(os.Stderr, "Input: ")
	} else {
		fmt.Fprintf(os.Stderr, prompt)
	}
	input := bufio.NewScanner(os.Stdin)
	input.Scan()
	return input.Text()
}

// no transform ...
func bash_no(cmdline string) string {
	return "{{bash \"" + cmdline + "\"}}"
}

func bash(cmdline string) string {
	cmd := exec.Command("/bin/bash", "-c", cmdline)
	out, err := cmd.Output()
	checkerror(err)

	return strings.Trim(string(out), "\n")
}

// no transform ...
func create_no(cmdline string) string {
	return "{{create \"" + cmdline + "\"}}"
}

func create(cmdline string) string {
	cmd := exec.Command("/bin/bash", "-c", "create_help " + cmdline)
	out, err := cmd.Output()
	checkerror(err)

	return strings.Trim(string(out), "\n")
}

func prepare(fname string) string {
	// Prepare input ...
	fmt.Fprintf(os.Stderr, "Input for template file '" + fname + "':\n")
	fmt.Fprintf(os.Stderr, "-------------------------------------\n")

	funcMap := template.FuncMap{"input": input, "bash" : bash_no, "create": create_no}

	// Template name
	tname := filepath.Base(fname)

	t, e := template.New(tname).Funcs(funcMap).ParseFiles(fname)
	checkerror(e)

	output := new(bytes.Buffer) 	// output is io.Writer

	e = t.Execute(output, value)
	checkerror(e)		

	return output.String()
}

func main() {
	flag.Parse()

	if len(fname) > 0 {
		text := prepare(fname)

		// Delete first blank line for help line, {{/*#help#: This is help.*/}}
		text = strings.Replace(text, "\n", "", 1)

		funcMap := template.FuncMap{"bash": bash, "create": create}

		// Template name
		tname := filepath.Base(fname)

		t, e := template.New(tname).Funcs(funcMap).Parse(text)
		checkerror(e)

		e = t.Execute(os.Stdout, nil)
		checkerror(e)		
	} else {
		usage()
	}
}
