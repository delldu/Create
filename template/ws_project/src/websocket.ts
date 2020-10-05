// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

function convertURIToImageData(URI: string) {
    return new Promise(function(resolve, reject) {
        if (URI == null)
            return reject();
        let canvas = document.createElement('canvas'),
            context = canvas.getContext('2d'),
            image = new Image();
        image.addEventListener('load', function() {
            canvas.width = image.width;
            canvas.height = image.height;
            if (context) {
                context.drawImage(image, 0, 0, canvas.width, canvas.height);
                resolve(context.getImageData(0, 0, canvas.width, canvas.height));
            }
        }, false);
        image.src = URI;
    });
}
let URI = "data:image/x-icon;base64,AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAQAQAABMLAAATCwAAAAAAAAAAAABsiqb/bIqm/2yKpv9siqb/bIqm/2yKpv9siqb/iKC3/2yKpv9siqb/bIqm/2yKpv9siqb/bIqm/2yKpv9siqb/bIqm/2yKpv9siqb/bIqm/2yKpv9siqb/2uLp///////R2uP/dZGs/2yKpv9siqb/bIqm/2yKpv9siqb/bIqm/2yKpv9siqb/bIqm/2yKpv9siqb/bIqm/////////////////+3w9P+IoLf/bIqm/2yKpv9siqb/bIqm/2yKpv9siqb/bIqm/2yKpv9siqb/bIqm/2yKpv///////////+3w9P+tvc3/dZGs/2yKpv9siqb/bIqm/2yKpv9siqb/TZbB/02Wwf9NlsH/TZbB/02Wwf9NlsH////////////0+Pv/erDR/02Wwf9NlsH/TZbB/02Wwf9NlsH/TZbB/02Wwf9NlsH/TZbB/02Wwf9NlsH/TZbB//////////////////////96sNH/TZbB/02Wwf9NlsH/TZbB/02Wwf9NlsH/TZbB/02Wwf9NlsH/TZbB/02Wwf////////////////+Ft9T/TZbB/02Wwf9NlsH/TZbB/02Wwf9NlsH/E4zV/xOM1f8TjNX/E4zV/yKT2P/T6ff/////////////////4fH6/z+i3f8TjNX/E4zV/xOM1f8TjNX/E4zV/xOM1f8TjNX/E4zV/xOM1f+m1O/////////////////////////////w+Pz/IpPY/xOM1f8TjNX/E4zV/xOM1f8TjNX/E4zV/xOM1f8TjNX////////////T6ff/Tqng/6bU7////////////3u/5/8TjNX/E4zV/xOM1f8TjNX/AIv//wCL//8Ai///AIv/////////////gMX//wCL//8gmv////////////+Axf//AIv//wCL//8Ai///AIv//wCL//8Ai///AIv//wCL///v+P///////+/4//+Axf//z+n/////////////YLf//wCL//8Ai///AIv//wCL//8Ai///AIv//wCL//8Ai///gMX/////////////////////////////z+n//wCL//8Ai///AIv//wCL//8Ai///AHr//wB6//8Aev//AHr//wB6//+Avf//7/f/////////////v97//xCC//8Aev//AHr//wB6//8Aev//AHr//wB6//8Aev//AHr//wB6//8Aev//AHr//wB6//8Aev//AHr//wB6//8Aev//AHr//wB6//8Aev//AHr//wB6//8Aev//AHr//wB6//8Aev//AHr//wB6//8Aev//AHr//wB6//8Aev//AHr//wB6//8Aev//AHr//wB6//8Aev//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==";
// convertURIToImageData(URI).then(function(imageData) {
//     // Here you can use imageData
//     console.log(imageData);
// });

// export class RotateImageFileProcessor implements ImageFileProcessor {
//     async process(dataURL: string): Promise < string > {
//         const canvas = document.createElement('canvas');
//         const image = await createImageFromDataUrl(dataURL);
//         canvas.width = image.height;
//         canvas.height = image.width;
//         const ctx = canvas.getContext("2d");
//         ctx.save();
//         ctx.translate(canvas.width / 2, canvas.height / 2);
//         ctx.rotate(Math.PI / 2);
//         ctx.drawImage(image, -image.width / 2, -image.height / 2);
//         ctx.restore();
//         return canvas.toDataURL(getImageTypeFromDataUrl(dataURL));
//     }
// }

class TensorHead {
    // CxHxW
    c: number; // 2 bytes
    h: number; // 2 bytes
    w: number; // 2 bytes
    opcode: number; // opcode, 1 byte

    constructor() {
        this.c = 0;
        this.h = 0;
        this.w = 0;
        this.opcode = 0;
    }

    setSize(c: number, h: number, w: number) {
        this.c = (c & 0xffff);
        this.h = (h & 0xffff);
        this.w = (w & 0xffff);
    }

    encode(): ArrayBuffer {
        let p = new ArrayBuffer(8);
        let c = new Uint8Array(p);
        let b = new Uint16Array(p);
        b[0] = this.c & 0xffff;
        b[1] = this.h & 0xffff;
        b[2] = this.w & 0xffff;
        c[6] = this.opcode & 0xff;
        c[7] = this.crc8(c, 7);

        // console.log("encode array:", c);
        // console.log("encode crc:", c[7]);
        return p;
    }

    decode(p: ArrayBuffer): boolean {
        let b = new Uint16Array(p);
        let c = new Uint8Array(p);

        this.c = b[0];
        this.h = b[1];
        this.w = b[2];
        this.opcode = c[6];

        // console.log("decode array:", c);
        // console.log("decode raw crc:", c[7]);
        // console.log("decode cal crc:", this.crc8(c, 7));
        return (c[7] == this.crc8(c, 7));
    }

    dataSize(): number {
        return this.c * this.h * this.w;
    }

    crc8(b: Uint8Array, n: number): number {
        let crc = 0;
        let odd;

        for (let i = 0; i < n; i++) {
            crc = crc ^ b[i];

            for (let j = 0; j < 8; j++) {
                odd = crc & 0x80;
                crc = crc << 1;
                if (odd) {
                    crc = crc ^ 0x07 % 256;
                } else {
                    crc = crc % 256;
                }
            }
        }

        return crc & 0xff;
    }
}

type TensorData = Uint8ClampedArray;

class Tensor {
    head: TensorHead;
    data: TensorData; // For RGBA, channel is 4 ...

    constructor() {
        this.head = new TensorHead();
        this.data = new Uint8ClampedArray(0);
    }

    encode(): ArrayBuffer {
        let h = new Uint8Array(this.head.encode());
        let p = new ArrayBuffer(8 + this.head.dataSize());
        let c1 = new Uint8Array(p);
        let c2 = new Uint8Array(p, 8);

        // copy head
        c1.set(h);
        // copy data
        c2.set(this.data);

        return p;
    }

    decode(p: ArrayBuffer): boolean {
        let ok = this.head.decode(p);
        if (ok && p.byteLength == (8 + this.head.dataSize())) {
            this.data = new Uint8ClampedArray(p, 8);
            return true;
        }
        return false;
    }

    valid(): boolean {
        return (this.head.dataSize() == this.data.length);
    }
}

function TensorPerformance() {
    let start_time = (new Date()).getTime();
    for (let i = 0; i < 1000; i++) {
        let x = new Tensor();
        x.head = new TensorHead();
        x.head.setSize(3, 2048, 4096);
        // h.setSize(3, 1024, 2048);
        x.head.opcode = 12;
        x.data = new Uint8ClampedArray(new ArrayBuffer(x.head.dataSize()));

        let p = x.encode();

        let y = new Tensor();
        let ok = y.decode(p);

        if (i % 100 == 0)
            console.log("i = ", i, ", encode ok:", x.valid(), ", decode ok:", ok);
    }
    let stop_time = (new Date()).getTime();
    console.log("Spend time: ", stop_time - start_time, "ms");
}

// TensorPerformance();



// console.log("h:", h);
// let e = h.encode();
// console.log("e:", e);
// let d = h.decode(e);
// console.log("d:", d, "h: ", h);

// let data = new TensorData(256, 256);
// console.log("data: ", data);
// let a = new Uint8ClampedArray(data.data);
// console.log("Uint8Array: ", a);
// for (let i = 0; i < a.length; i++) {
//     a[i] = data.data[i];
// }

// for (let i = 0; i < data.data.length; i++) {
//     a[i] = data.data[i];
// }


// var blob = new Blob(["Hello World"]);
// var wsurl = window.URL.createObjectURL(blob);
// var a = document.getElementById("h");
// a.download = "helloworld.txt";
// a.href = wsurl;
//   

// let img = new Image();
// img.src = URI;
// console.log("What's is image? ", img);

class TensorClient {
    wsurl: string;
    socket: WebSocket;

    constructor(wsurl: string) {
        this.wsurl = wsurl;
        this.socket = new WebSocket(wsurl);
        // this.registerHandlers();
    }

    registerHandlers() {
        if (!this.socket)
            return;

        // Register event message ?
        this.socket.addEventListener('open', (event: Event) => {
            console.log("WebSocket open on " + this.socket.url + " ...");
        }, false);

        this.socket.addEventListener('close', (event: CloseEvent) => {
            // var code = event.code;
            // var reason = event.reason;
            // var wasClean = event.wasClean;
            // handle close event
            console.log("WebSocket closed for " + event.reason + "(" + event.code + ")");
        }, false);

        this.socket.binaryType = "arraybuffer";
    }

    echo_start(x: Tensor) {
        return new Promise((resolve: (value: ArrayBuffer) => void, reject: (value: string) => void) => {
            if (!x.valid()) {
                return reject("Invalid input tensor.");
            }

            this.socket.addEventListener('message', (event: MessageEvent) => {
                if (event.data instanceof String) {
                    console.log("Received data string");
                }

                if (event.data instanceof ArrayBuffer) {
                    console.log("Received arraybuffer");
                    resolve(event.data);
                }
            }, false);

            this.socket.addEventListener('error', (event: Event) => {
                return reject("WebSocket open error.")
                // handle error event
            }, false);

            if (this.socket.readyState != WebSocket.OPEN) {
                return reject("WebSocket not opened.");
            }

            this.socket.send(x.encode());
        });
    }

    echo_stop(x: Tensor): [boolean, Tensor] {
        let ok = false;
        let y = new Tensor();
        this.echo_start(x).then(
            (buffer: ArrayBuffer) => {
                // receive is valid tensor ?
                if (y.decode(buffer))
                    ok = true;
            },
            (err_reason: string) => {
                ok = false;
            });
        return [ok, y];
    }

    close() {
        this.socket.close();
    }
}

let client = new TensorClient("socket://localhost:8080");



// let wsurl = "socket://localhost:8080";



// socket.readyState == WebSocket.OPEN
// socket.binaryType = "arraybuffer";
// socket.send(buffer);

// socket.onmessage = function(event) {
//     if (event.data instanceof String) {
//         console.log("Received data string");
//     }

//     if (event.data instanceof ArrayBuffer) {
//         var buffer = event.data;
//         console.log("Received arraybuffer");
//     }

//     console.log("Received Message: " + event.data);
// };

// socket.onerror = function(event) {
//     // handle error event
// };