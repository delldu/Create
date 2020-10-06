// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

// Decode dataURL OK ?
let startDataURLToImage = (url: string) => {
    return new Promise(function(resolve, reject) {
        if (url == null)
            return reject();
        let image = new Image();
        image.addEventListener('load', function() {
            resolve();
        }, false);
        image.addEventListener('abort', function() {
            reject();
        }, false);
        image.addEventListener('error', function() {
            reject();
        }, false);
        image.src = url;
    });
}

// convertURIToImageData(URI).then(function(imageData) {
//     // Here you can use imageData
//     console.log(imageData);
// });

class ImageProjectItem {
    name: string;
    size: number;
    dataurl: string; // RFC 2397
    blobs: string;

    constructor(name: string, size: number, dataurl: string, blobs: string) {
        this.name = name;
        this.size = size;
        this.dataurl = dataurl;
        this.blobs = blobs;
    }
}

class ImageProject {
    name: string;
    create: Date;
    items: Array < ImageProjectItem > ;

    decode_ok_count: number;
    decode_err_count: number;

    constructor(name: string) {
        this.name = name;
        this.create = new Date();
        this.items = new Array < ImageProjectItem > ();

        // Statics
        this.decode_ok_count = 0;
        this.decode_err_count = 0;
    }

    load(file: File) {
        if (file instanceof File) {
            let reader = new FileReader();
            reader.addEventListener("error", () => {
                this.decode_err_count++;
            }, false);
            reader.addEventListener("load", () => {
                // reader.result ok ?
                startDataURLToImage(reader.result).then(() => {
                    let unit = new ImageProjectItem(file.name, file.size, reader.result, "");
                    this.items.push(unit);
                    this.decode_ok_count++;
                }, () => {
                    this.decode_err_count++;
                });
            }, false);
            reader.readAsDataURL(file);
        } else {
            this.decode_err_count++;
        }
    }

    show(index: number, id: string) {
        // Wait loading finish ?
        setTimeout(() => {
            let e = document.getElementById(id) as HTMLImageElement;
            if (e)
                e.src = this.items[index].dataurl;
        }, 20); // 20 ms is reasonable for human bings
    }

    image(index: number): HTMLImageElement {
        let image = new Image();
        image.src = this.items[index].dataurl;
        return image as HTMLImageElement;
    }

    // JSON string
    json(): string {
        return "";
    }

    listHtml(): string {
        return "";
    }

    gridHtml(): string {
        return "";
    }

    // JSON format file
    open(file: File) {

    }

    // JSON format
    save(filename: string) {

    }

    info(): string {
        let decode_total = this.decode_ok_count + this.decode_err_count;
        return "Project name: " + this.name +
            ", create time: " + this.create +
            ", decode " + this.decode_ok_count + " OK" +
            ", " + this.decode_err_count + " error" +
            ", total: " + decode_total + ".";
    }
}

let project = new ImageProject("Demo");
console.log(project.info());


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

enum NImageOpcode {
    Clean = 2,
        Zoom = 4,
        Color = 6,
        Patch = 8
};

class NImageHead {
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

type NImageData = Uint8ClampedArray;

class NImage {
    head: NImageHead;
    data: NImageData; // For RGBA, channel is 4 ...

    constructor() {
        this.head = new NImageHead();
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

function NImagePerformance() {
    let start_time = (new Date()).getTime();
    for (let i = 0; i < 1000; i++) {
        let x = new NImage();
        x.head = new NImageHead();
        x.head.setSize(3, 2048, 4096);
        // h.setSize(3, 1024, 2048);
        x.head.opcode = 12;
        x.data = new Uint8ClampedArray(new ArrayBuffer(x.head.dataSize()));

        let p = x.encode();

        let y = new NImage();
        let ok = y.decode(p);

        if (i % 100 == 0)
            console.log("i = ", i, ", encode ok:", x.valid(), ", decode ok:", ok);
    }
    let stop_time = (new Date()).getTime();
    console.log("Spend time: ", stop_time - start_time, "ms");
}

// NImagePerformance();



// console.log("h:", h);
// let e = h.encode();
// console.log("e:", e);
// let d = h.decode(e);
// console.log("d:", d, "h: ", h);

// let data = new NImageData(256, 256);
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

class NImageClient {
    wsurl: string;
    socket: WebSocket;

    constructor(wsurl: string) {
        this.wsurl = wsurl;
        this.socket = new WebSocket(wsurl);

        this.registerHandlers();
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

    private echo_start(x: NImage) {
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

    // Call echo service via websocket
    echoService(x: NImage): [boolean, NImage] {
        let ok = false;
        let y = new NImage();
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

    clean(x: NImage): [boolean, NImage] {
        x.head.opcode = NImageOpcode.Clean;
        return this.echoService(x);
    }

    zoom(x: NImage): [boolean, NImage] {
        x.head.opcode = NImageOpcode.Zoom;
        return this.echoService(x);
    }

    color(x: NImage): [boolean, NImage] {
        x.head.opcode = NImageOpcode.Color;
        return this.echoService(x);
    }

    patch(x: NImage): [boolean, NImage] {
        x.head.opcode = NImageOpcode.Patch;
        return this.echoService(x);
    }

    close() {
        this.socket.close();
    }
}

// let client = new NImageClient("socket://localhost:8080");



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