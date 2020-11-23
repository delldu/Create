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

void fatal(const char *message, int errcode)
{
	syslog_error("%s: %s\n", message, nng_strerror(errcode));

	exit(-1);
}

int slow_send_image(nng_socket socket, IMAGE *image)
{
	int ret;
	BYTE *send_buf;
	size_t send_size;

	send_buf = image_encode(image, &send_size);
	if ((ret = nng_send(socket, send_buf, send_size, 0)) != 0) {
		fatal("nng_send", ret);
	}
	free(send_buf); // Send finish, so release buffer
	
	return RET_OK;
}

int fast_send_image(nng_socket socket, IMAGE *image)
{
	int ret;
	nng_msg *msg = NULL;

	BYTE head_buf[64];
	size_t send_size;

	image_head_encode(image, head_buf, &send_size);
	if ((ret = nng_msg_alloc(&msg, 0)) != 0) {
		fatal("nng_msg_alloc", ret);
	}
	if ((ret = nng_msg_append(msg, head_buf, send_size)) != 0) {
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

IMAGE *pix2pix_service(IMAGE *image)
{
	CHECK_IMAGE(image);
	
	// Transform picture ...
	// color_togray(image);

	return image_copy(image);
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

	IMAGE *recv_image, *send_image = NULL;
	BYTE *recv_buf;
	size_t recv_size;

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

		recv_image = image_decode(recv_buf);
		nng_free(recv_buf, recv_size);	// Data has been saved to recv_image ...
		if (! image_valid(recv_image)) {
			continue;
		}

		// Do some intresting thing ...
		// if (recv_image->opc == 0x1234)
		// 	color_togray(recv_image);
		// send_image = image_copy(recv_image);

		if (send_image)	// Keep for fast_send_image finished !!!
			image_destroy(send_image);

		send_image = pix2pix_service(recv_image);
#ifdef SEND_FAST
	    fast_send_image(socket, send_image);
#else
	    slow_send_image(socket, send_image);
#endif
		image_destroy(recv_image);
	}

	syslog(LOG_INFO, "Image service shutdown.\n");
	nng_close(socket);

	return RET_OK;
}

IMAGE *run_pix2pix_service(nng_socket socket, IMAGE *send_image)
{
	int ret;
	IMAGE *recv_image;
	BYTE *recv_buf = NULL;
	size_t recv_size;

	CHECK_IMAGE(send_image);

	// Send ...
#ifdef SEND_FAST
	    fast_send_image(socket, send_image);
#else
	    slow_send_image(socket, send_image);
#endif
	// Receive ...
	if ((ret = nng_recv(socket, &recv_buf, &recv_size, NNG_FLAG_ALLOC)) != 0) {
		fatal("nng_recv", ret);
	}
	recv_image = image_decode(recv_buf); CHECK_IMAGE(recv_image);
	nng_free(recv_buf, recv_size); // recv_buf bee saved to recv_image, so release recv_buf ...

	return recv_image;
}

int client(char *input_file, char *cmd, char *output_file)
{
	int ret;
	nng_socket socket;
	IMAGE *send_image, *recv_image;

	if ((ret = nng_req0_open(&socket)) != 0) {
		fatal("nng_socket", ret);
	}
	if ((ret = nng_dial(socket, URL, NULL, 0)) != 0) {
		fatal("nng_dial", ret);
	}
	send_image = image_load(input_file); check_image(send_image);

	// Test performance
	int k;
	time_reset();
	printf("Test image service performance ...\n");
	for (k = 0; k < 100; k++) {
		printf("%d ...\n", k);
		recv_image = run_pix2pix_service(socket, send_image); check_image(recv_image);
		image_destroy(recv_image);
	}
	time_spend("Image service 100 times");

	recv_image = run_pix2pix_service(socket, send_image); check_image(recv_image);
	image_save(recv_image, output_file);
	image_destroy(recv_image);

	image_destroy(send_image);

	nng_close(socket);

	return RET_OK;
}

void help(char *cmd)
{
	printf("Usage: %s [option]\n", cmd);
	printf("    -h, --help                   Display this help.\n");
	printf("    -s, --server                 Start server.\n");
	printf("    -c, --client --input <file> --execute <clean|color|zoom|patch> --output <file>\n");
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

	if (run_client && input_file && command && output_file) {
		return client(input_file, command, output_file);
	}

	// error ？
	help(argv[0]);

	return RET_ERROR;
}
