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

#include <nng/nng.h>
#include <nng/protocol/reqrep0/rep.h>
#include <nng/protocol/reqrep0/req.h>

#include <nimage/image.h>

#define URL "ipc:///tmp/image.ipc"

#define SEND_FAST 1

// nng/nng.h define ...
// typedef struct nng_socket_s {
//         uint32_t id;
// } nng_socket;

// int image_data_size(IMAGE *image);
// int image_abhead_encode(IMAGE *image, BYTE *buffer);
// int image_abhead_decode(BYTE *buffer, AbHead *abhead);


void fatal(const char *message, int errcode)
{
	syslog_error("%s: %s\n", message, nng_strerror(errcode));

	exit(-1);
}

int fast_send_image(nng_socket socket, IMAGE *image)
{
	int ret;
	nng_msg *msg = NULL;

	BYTE head_buf[sizeof(AbHead)];
	size_t send_size;

	image_abhead_encode(image, head_buf);
	if ((ret = nng_msg_alloc(&msg, 0)) != 0) {
		fatal("nng_msg_alloc", ret);
	}
	if ((ret = nng_msg_append(msg, head_buf, sizeof(AbHead))) != 0) {
		fatal("nng_msg_append", ret);
	}
	send_size = image_data_size(image);
	if ((ret = nng_msg_append(msg, image->base, send_size)) != 0) {
		fatal("nng_msg_append", ret);
	}
	if ((ret = nng_sendmsg(socket, msg, 0)) != 0) {
		fatal("nng_sendmsg", ret);
	}
	nng_msg_free(msg);

	return RET_OK;	
}

BYTE *pix2pix_service(BYTE *recv_buf, size_t recv_size, size_t *send_size)
{
	int ret;
	AbHead abhead;
	BYTE *send_buf;

	// Check recv_buf is a image array buffer ?
	ret = image_abhead_decode(recv_buf, &abhead);
	if (ret != RET_OK || abhead.len + sizeof(AbHead) != recv_size) {
		syslog_error("Bad ArrayBuffer data.");
		return NULL;
	}
	// demo: echo service

	*send_size = recv_size;
	send_buf = (BYTE *)malloc(recv_size);
	if (send_buf) {
		memcpy(send_buf, recv_buf, recv_size);
	}
	return send_buf;
}

BYTE *pix2txt_service(IMAGE *image)
{
	BYTE *response;
	CHECK_IMAGE(image);

	response = (BYTE *)malloc(1024);
	if (response) {
		memset(response, 0, 1024);
		snprintf((char *)response, 1024, "Hello, Image !");
	}

	return response;
}

// start_pix2pix_server
// start_pix2txt_server
int server()
{
	int ret;
	nng_socket socket;
	BYTE *recv_buf, *send_buf = NULL;
	size_t recv_size, send_size;

	// sudo journalctl -u image.service -n 10
	syslog(LOG_INFO, "Start image service on %s ...\n", URL);
	if ((ret = nng_rep0_open(&socket)) != 0) {
		fatal("nng_rep0_open", ret);
	}
	if ((ret = nng_listen(socket, URL, NULL, 0)) != 0) {
		fatal("nng_listen", ret);
	}
	syslog(LOG_INFO, "Image service already ...\n");

	for (;;) {
		recv_buf = NULL;
		if ((ret = nng_recv(socket, &recv_buf, &recv_size, NNG_FLAG_ALLOC)) != 0) {
			fatal("nng_recv", ret);
		}

		if (send_buf)	// Keep for fast_send_image finished !!!
			free(send_buf);

		send_buf = pix2pix_service(recv_buf, recv_size, &send_size);
		if ((ret = nng_send(socket, send_buf, send_size, 0)) != 0) {
			fatal("nng_send", ret);
		}
		nng_free(recv_buf, recv_size);	// Data has been saved to recv_image ...
	}

	syslog(LOG_INFO, "Image service shutdown.\n");
	nng_close(socket);

	return RET_OK;
}

int run_pix2pix_service(nng_socket socket, IMAGE *send_image)
{
	int ret;
	BYTE *recv_buf = NULL;
	size_t recv_size;

	check_image(send_image);

	// Send ...
    fast_send_image(socket, send_image);

	// Receive ...
	if ((ret = nng_recv(socket, &recv_buf, &recv_size, NNG_FLAG_ALLOC)) != 0) {
		fatal("nng_recv", ret);
	}
	nng_free(recv_buf, recv_size); // release recv_buf ...

	return RET_OK;
}

int client(char *input_file, char *cmd)
{
	int ret;
	nng_socket socket;
	IMAGE *send_image;

	if ((ret = nng_req0_open(&socket)) != 0) {
		fatal("nng_socket", ret);
	}
	if ((ret = nng_dial(socket, URL, NULL, 0)) != 0) {
		fatal("nng_dial", ret);
	}
	send_image = image_load(input_file); check_image(send_image);
	
	if (cmd)
		send_image->opc = 0x1234;

	// Test performance
	int k;
	time_reset();
	printf("Test image service performance ...\n");
	for (k = 0; k < 100; k++) {
		printf("%d ...\n", k);
		run_pix2pix_service(socket, send_image);
	}
	time_spend("Image service 100 times");

	run_pix2pix_service(socket, send_image);

	image_destroy(send_image);

	nng_close(socket);

	return RET_OK;
}

void help(char *cmd)
{
	printf("Usage: %s [option]\n", cmd);
	printf("    -h, --help                   Display this help.\n");
	printf("    -s, --server                 Start server.\n");
	printf("    -c, --client --input <file> --execute <clean|color|zoom|patch>\n");
	printf("                                 Start client.\n");

	exit(1);
}

int main(int argc, char **argv)
{
	int optc;
	int option_index = 0;
	int run_client = 0;	// 0 -- server
	char *input_file = NULL;
	char *command = NULL;

	struct option long_opts[] = {
		{ "help", 0, 0, 'h'},
		{ "server", 0, 0, 's'},
		{ "client", 0, 0, 'c'},
		{ "input", 1, 0, 'i'},
		{ "execute", 1, 0, 'e'},
		{ 0, 0, 0, 0}
	};

	if (argc <= 1)
		help(argv[0]);
	
	while ((optc = getopt_long(argc, argv, "h s c i: e:", long_opts, &option_index)) != EOF) {
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
		case 'h':	// help
		default:
			help(argv[0]);
			break;
	    }
	}

	if (run_client && input_file && command) {
		return client(input_file, command);
	}

	// error ？
	help(argv[0]);

	return RET_ERROR;
}
