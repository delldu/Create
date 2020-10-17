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



// wsab_start, wsab_stop, ws + ab
const ab_stop = function(time: number): Promise < number > {
    return new Promise(function(resolve, reject) {
        setTimeout(function() {
            resolve(1024);
        }, time);
    })
};

async function ab_start() {
    console.log('start');
    let result = await ab_stop(1000);
    console.log(result);

    console.log('end');
};

ab_start();


// const demo = async (ws: WsClient, x: ArrayBuffer) => {
//     await ws.open();
//     const y = await ws.send(x);
// }

// let ws = new WsClient("ws://127.0.0.1:8080");
// demo(ws, x);
// ws.close();

// WebSocket.CONNECTING    0
// WebSocket.OPEN  1
// WebSocket.CLOSING   2
// WebSocket.CLOSED 3



const DEFAULT_RECONNECT_INTERVAL = 5000; // 5s

class AbClient {
    private address: string;
    private socket: any; // WebSocket;
    private status: number;
    private timer: number; // Timer

    constructor(address: string) {
        this.address = address;

        this.status = WebSocket.CLOSED;
        this.timer = setInterval(() => {
            this.open();
        }, DEFAULT_RECONNECT_INTERVAL);
        this.socket = null;
    }

    open() {
        if (this.status == WebSocket.CONNECTING || this.status == WebSocket.OPEN)
            return;

        this.socket = new WebSocket(this.address);
        this.socket.binaryType = "arraybuffer";

        this.socket.addEventListener('open', (event: Event) => {
            console.log("WebSocket open on " + this.socket.url + " ...");
            this.status = WebSocket.OPEN;
        }, false);

        this.socket.addEventListener('close', (event: CloseEvent) => {
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