// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

class Message {
    timer: number;
    element: HTMLElement; // message element

    constructor(id: string) {
        this.timer = 0;
        this.element = document.getElementById(id) as HTMLElement;
        this.element.addEventListener('dblclick', function() {
            this.style.display = 'none';
        }, false);

        console.log("Message: double click hidden messgae.");
    }

    // show message for t micro seconds
    show(msg: string, t: number) {
        if (this.timer) {
            clearTimeout(this.timer);
        }
        this.element.innerHTML = msg;
        this.element.style.display = ""; // Show
        this.timer = setTimeout(() => {
            this.element.style.display = 'none';
        }, t);
    }
}

// Test:
// <div id="message_id" class="message">Message Bar</div>
// function load() {
//     msgbar = new Message("message_id");
//     msgbar.show("This is a message ........................", 10000); // 10 s
// }

const DEFAULT_RECONNECT_INTERVAL = 30*1000; // 30s

class AbClient {
    private address: string;
    private socket: any; // WebSocket;
    private status: number;
    private timer: number; // Timer

    constructor(address: string) {
        this.address = address;
        this.socket = null;
        // Define status for socket.readyState could not be used(socket == null)
        this.status = WebSocket.CLOSED;
        this.timer = 0; // Re-connect timer

        this.open();
    }

    open() {
        // console.log("Start re-connect timer...");
        if (this.timer <= 0) {
            this.timer = setInterval(() => {
                this.open();
            }, DEFAULT_RECONNECT_INTERVAL);
        }

        if (this.status == WebSocket.CONNECTING || this.status == WebSocket.OPEN) {
            console.log("WebSocket is going on ...");
            return;
        }

        this.socket = new WebSocket(this.address);
        this.socket.binaryType = "arraybuffer";

        this.socket.addEventListener('open', (event: Event) => {
            console.log("WebSocket open on " + this.socket.url + " ...");
            this.status = WebSocket.OPEN;
        }, false);

        this.socket.addEventListener('close', (event: CloseEvent) => {
            this.status = WebSocket.CLOSING;
            console.log("WebSocket closed for " + event.reason + "(" + event.code + ")");
            this.status = WebSocket.CLOSED;
        }, false);
    }

    send(ablist: Array < ArrayBuffer > ): Promise < ArrayBuffer > {
        return new Promise((resolve, reject) => {
            if (this.status != WebSocket.OPEN) {
                reject("WebSocket not opened.");
            }

            this.socket.addEventListener('message', (event: MessageEvent) => {
                if (event.data instanceof String) {
                    console.log("Received string data ... ", event.data);
                }
                if (event.data instanceof ArrayBuffer) {
                    console.log("Received ArrayBuffer ... ", event.data);
                    resolve(event.data);
                }
            }, false);

            this.socket.addEventListener('error', (event: Event) => {
                reject("WebSocket error.")
            }, false);

            // Send all data in the list of arraybuffer
            for (let x of ablist) {
                this.socket.send(x);
            }
        });
    }

    close() {
        this.status = WebSocket.CLOSING;
        if (this.timer)
            window.clearInterval(this.timer);
        this.timer = 0;

        this.socket.close(1000);
        this.status = WebSocket.CLOSED;
    }
}

// Demo:
// new AbClient("ws://locahost:8080").send([a, b, c])
// .then((ab:ArrayBuffer)=>{ console.log(ab); })
// .catch((error)=>{ console.log("error."); });