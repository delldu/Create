/************************************************************************************
***
***	Copyright 2019 Dell(18588220928@163.com), All Rights Reserved.
***
***	File Author: Dell, 2019-08-08 23:02:07
***
************************************************************************************/

package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"strings"
	"os"
	"os/exec"
    "os/user"
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

	fmt.Println("create_from_template version:", version)
	fmt.Println("Usage: create_from_template [-f file] [-v value] [-h]")
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

func runcmd(cmdline string) string {
	cmd := exec.Command("/bin/bash", "-c", cmdline)
	out, err := cmd.Output()
	checkerror(err)

	return strings.Trim(string(out), "\n")
}

func whoami() string {
	u, err := user.Current()
	checkerror(err)
	b := []rune(u.Username)
	if b[0] >= 97 && b[0] <= 122 {
		b[0] -= 32
	} 
	return string(b)
}

func main() {
	flag.Parse()

	if len(fname) > 0 {
		b, e := ioutil.ReadFile(fname)
		checkerror(e)
		text := string(b)

		funcMap := template.FuncMap{"sh": runcmd, "bash": runcmd, "who": whoami}

		// Template name
		tname := filepath.Base(fname)

		t, e := template.New(tname).Funcs(funcMap).Parse(text)
		checkerror(e)

		e = t.Execute(os.Stdout, value)
	} else {
		usage()
	}
}
