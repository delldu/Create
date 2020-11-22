/************************************************************************************
***
***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2020-11-22 13:18:11
***
************************************************************************************/


#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <syslog.h>

#define RET_OK 0
#define RET_ERROR (-1)

#define URL "ipc:///tmp/image.socket"

int server()
{
	syslog(LOG_INFO, "Satrt image service on %s ...\n", URL);

	while(1) {
		syslog(LOG_INFO, "Running image service ...\n");
		sleep(5);
	}

	return RET_OK;
}

int client(char *input_file, char *cmd, char *outfile)
{
	printf("Connect %s\n", URL);
	printf("%s %s to %s\n", cmd, input_file, outfile);

	return RET_OK;
}

void help(char *cmd)
{
	printf("Usage: %s [option]\n", cmd);
	printf("    -h, --help                   Display this help.\n");
	printf("    -s, --server                 Start server.\n");
	printf("    -c, --client --input <image> \
					--execute <clean|color|zoom|patch> \
					--output <image> \
										     Start client.\n");

	exit(1);
}

int main(int argc, char **argv)
{
	int optc;
	int option_index = 0;
	int run_client = 0;	// 0 -- server
	char *input_file = NULL;
	char *command = NULL;
	char *output_file = NULL;

	struct option long_opts[] = {
		{ "help", 0, 0, 'h'},
		{ "server", 0, 0, 's'},
		{ "client", 0, 0, 'c'},
		{ "input", 1, 0, 'i'},
		{ "execute", 1, 0, 'e'},
		{ "output", 1, 0, 'o'},
		{ 0, 0, 0, 0}
	};

	if (argc <= 1)
		help(argv[0]);
	
	while ((optc = getopt_long(argc, argv, "h s c i: e: o:", long_opts, &option_index)) != EOF) {
		switch (optc) {
		case 's':
			run_client = 0;
			return server();
			break;
		case 'c':
			run_client = 1;
			break;
		case 'i':
			input_file = optarg;
			break;
		case 'e':
			command = optarg;
			break;
		case 'o':
			output_file = optarg;
			break;
		case 'h':	// help
		default:
			help(argv[0]);
			break;
	    }
	}

	if (run_client && ! input_file && ! command && ! output_file)
		return client(input_file, command, output_file);

	// error ？
	help(argv[0]);

	return RET_ERROR;
}
