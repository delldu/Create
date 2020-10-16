// Copyright 2015 The Gorilla WebSocket Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"time"

	"github.com/gorilla/websocket"
)

const (
	// Time allowed to write a message to the peer.
	writeWait = 10 * time.Second

	// Maximum message size allowed from peer.
	maxMessageSize = 24 * 1024 * 1024

	// Time allowed to read the next pong message from the peer.
	pongWait = 60 * time.Second

	// Send pings to peer with this period. Must be less than pongWait.
	pingPeriod = (pongWait * 9) / 10

	// Time to wait before force close on connection.
	closeGracePeriod = 10 * time.Second
)

func pumpStdin(ws *websocket.Conn, w io.Writer) {
	defer ws.Close()
	ws.SetReadLimit(maxMessageSize)
	ws.SetReadDeadline(time.Now().Add(pongWait))
	ws.SetPongHandler(func(string) error {
		ws.SetReadDeadline(time.Now().Add(pongWait))
		return nil
	})
	for {
		_, message, err := ws.ReadMessage()
		if err != nil {
			break
		}
		// message = append(message, '\n') 	// BinaryMessage donot append '\n' !!!
		if _, err := w.Write(message); err != nil {
			break
		}
	}
}

func pumpStdout(ws *websocket.Conn, r io.Reader, done chan struct{}) {
	// s := bufio.NewScanner(r)
	// for s.Scan() {
	// 	ws.SetWriteDeadline(time.Now().Add(writeWait))
	// 	// if err := ws.WriteMessage(websocket.TextMessage, s.Bytes()); err != nil {
	// 		ws.Close()
	// 		break
	// 	}
	// }
	// if s.Err() != nil {
	// 	log.Println("Scan error:", s.Err())
	// }
	s := bufio.NewReader(r)
	for {
		buf := make([]byte, 4*1024*1024)
		n, err := s.Read(buf)
		if err != nil {
			if err == io.EOF {
				break;
			} else {
				log.Println("Reader error.");
			}
		} else {
			buf = buf[:n]
			ws.SetWriteDeadline(time.Now().Add(writeWait))
			if err := ws.WriteMessage(websocket.BinaryMessage, buf); err != nil {
				ws.Close()
				break
			}
		}
	}

	close(done)

	ws.SetWriteDeadline(time.Now().Add(writeWait))
	ws.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
	time.Sleep(closeGracePeriod)
	ws.Close()
}

func ping(ws *websocket.Conn, done chan struct{}) {
	ticker := time.NewTicker(pingPeriod)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			if err := ws.WriteControl(websocket.PingMessage, []byte{}, time.Now().Add(writeWait)); err != nil {
				log.Println("ping:", err)
			}
		case <-done:
			return
		}
	}
}

func websocketError(ws *websocket.Conn, msg string, err error) {
	log.Println(msg, err)
	ws.WriteMessage(websocket.TextMessage, []byte("WebSocket internal server error."))
}

func handleWebSocketService(w http.ResponseWriter, r *http.Request) {
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true
		},
	}

	ws, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("upgrade:", err)
		return
	}
	defer ws.Close()

	outr, outw, err := os.Pipe()
	if err != nil {
		websocketError(ws, "stdout:", err)
		return
	}
	defer outr.Close()
	defer outw.Close()

	inr, inw, err := os.Pipe()
	if err != nil {
		websocketError(ws, "stdin:", err)
		return
	}
	defer inr.Close()
	defer inw.Close()

	proc, err := os.StartProcess(command, flag.Args(), &os.ProcAttr{
		Files: []*os.File{inr, outw, outw},
	})
	if err != nil {
		websocketError(ws, "start:", err)
		return
	}
	inr.Close()
	outw.Close()

	stdoutDone := make(chan struct{})
	go pumpStdout(ws, outr, stdoutDone)
	go ping(ws, stdoutDone)

	pumpStdin(ws, inw)

	// Some commands will exit when stdin is closed.
	inw.Close()

	// Other commands need a bonk on the head.
	if err := proc.Signal(os.Interrupt); err != nil {
		log.Println("inter:", err)
	}

	select {
	case <-stdoutDone:
	case <-time.After(time.Second):
		// A bigger bonk on the head.
		if err := proc.Signal(os.Kill); err != nil {
			log.Println("term:", err)
		}
		<-stdoutDone
	}

	if _, err := proc.Wait(); err != nil {
		log.Println("wait:", err)
	}
}

func handleStdHttpService(w http.ResponseWriter, r *http.Request) {
	http.FileServer(http.Dir(".")).ServeHTTP(w, r)
}

func uploadError(w http.ResponseWriter, message string, statusCode int) {
	log.Println(message)
	w.WriteHeader(statusCode)
	w.Write([]byte(message))
}

func handleUploadService(w http.ResponseWriter, r *http.Request) {
	// Test command:
	// curl  -F "filename=@/home/test/file.tar.gz" http://127.0.0.1:8080/upload

	if r.Method == "POST" {
		f, h, err := r.FormFile("filename")
		if err != nil {
			uploadError(w, "Invalid file id", http.StatusBadRequest)
			return
		}
		defer f.Close()
		// validate file size
		if h.Size > 1024*1024*1024 {
			uploadError(w, "Upload file size > 1G.", http.StatusBadRequest)
			return
		}

		dstfilename := "/tmp/" + h.Filename
		t, err := os.Create(dstfilename)
		if err != nil {
			uploadError(w, "Create write file error.", http.StatusInternalServerError)
			return
		}
		defer t.Close()
		_, err = io.Copy(t, f)
		fmt.Fprintln(w, "Upload Success.")
	} else {
		uploadError(w, "Upload ONLY ACCEPT 'POST' request.", http.StatusBadRequest)
	}
}

var (
	help bool

	address string
	command string
)

func init() {
	flag.BoolVar(&help, "h", false, "Display this help")
	flag.StringVar(&address, "e", "0.0.0.0:8080", "Websocket service endpoint")

	// flag.PrintDefaults() is not good enough !
	flag.Usage = usage
}

func usage() {
	const version = "1.0"

	fmt.Println("Websocket Version:", version)
	fmt.Println("Usage: websocket [options] command")
	fmt.Println("Options:")

	fmt.Println("    -h               Display this help")
	fmt.Println("    -e address       Websocket service endpoint (default is 0.0.0.0:8080)")
}

func main() {
	flag.Parse()

	if len(flag.Args()) < 1 {
		usage()
		return
	}

	var err error
	command, err = exec.LookPath(flag.Args()[0])
	if err != nil {
		log.Fatal(err)
	}

	http.HandleFunc("/", handleStdHttpService)
	http.HandleFunc("/upload", handleUploadService)
	http.HandleFunc("/ws", handleWebSocketService)

	log.Printf("Starting websocket server at %s with command '%s' ...\n", address, command)
	log.Fatal(http.ListenAndServe(address, nil))
}
