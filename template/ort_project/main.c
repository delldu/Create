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

void help(char *cmd)
{
	printf("Usage: %s [option]\n", cmd);
	printf("    -h, --help                   Display this help.\n");

	exit(1);
}

int main(int argc, char **argv)
{
	int optc;
	int option_index = 0;

	struct option long_opts[] = {
		{"help", 0, 0, 'h'},
		{0, 0, 0, 0}

	};

	while ((optc = getopt_long(argc, argv, "h", long_opts, &option_index)) != EOF) {
		switch (optc) {
		case 'h':				// help
		default:
			help(argv[0]);
			break;
		}
	}

	EngineTest();

	return 0;
}
