// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

// dataURL specification: RFC 2397

const sleep = (time: number) => {
    return new Promise(resolve => setTimeout(resolve, time));
}
// sleep(2000).then(() => {
//     console.log("2 seconds passed.");
// })

const crc16 = (b: Uint8Array, n: number): number => {
    let crc = 0;
    let CRC_CCITT_POLY = 0x1021;
    for (let i = 0; i < n; i++) {
        crc = crc ^ (b[i] << 8);
        for (let j = 0; j < 8; j++) {
            if (crc & 0x8000)
                crc = (crc << 1) ^ CRC_CCITT_POLY;
            else
                crc = crc << 1;
        }
    }
    return crc & 0xffff;
}

// Decode dataURL to HTMLImageElement
function dataURLToImage(url: string): Promise < HTMLImageElement > {
    return new Promise(function(resolve, reject) {
        // Promise excutor ...
        if (url == null) {
            reject("dataURLToImage: url == null");
        }
        let e = new Image() as HTMLImageElement;
        e.addEventListener('load', () => {
            resolve(e);
        }, false);
        e.addEventListener('abort', () => {
            reject("dataURLToImage: abort");
        }, false);
        e.addEventListener('error', () => {
            reject("dataURLToImage: error");
        }, false);
        e.src = url;
    });
}

// Convert dataURL to ImageData (ArrayBuffer)
function dataURLToImageData(url: string): Promise < [number, number, ImageData] > {
    return new Promise(function(resolve, reject) {
        if (url == null)
            reject("dataURLToImageData: url == null");
        let canvas = document.createElement('canvas'),
            context = canvas.getContext('2d'),
            image = new Image();
        image.addEventListener('load', function() {
            canvas.width = image.width;
            canvas.height = image.height;
            if (context) {
                context.drawImage(image, 0, 0, canvas.width, canvas.height);
                resolve([canvas.height, canvas.width, context.getImageData(0, 0, canvas.width, canvas.height)]);
            }
        }, false);
        image.addEventListener('load', function() {
            reject("dataURLToImageData: error.");
        }, false);
        image.src = url;
    });
}

// dataURLToImageData(URI).then((imageData) => {
//     // Here you can use ImageData
//     console.log(imageData);
// });

// Load dataURL from file
function loadDataURLFromFile(file: File): Promise < string > {
    return new Promise(function(resolve, reject) {
        if (!(file instanceof File))
            reject("loadDataURLFromFile: input is not File object.");
        else {
            let reader = new FileReader();
            reader.addEventListener("error", () => {
                reject("loadDataURLFromFile: file read error.");
            }, false);
            reader.addEventListener("load", () => {
                // reader ok ?
                resolve(reader.result as string); // throw dataURL data
            }, false);
            reader.readAsDataURL(file);
        }
    });
}

class ImageProjectItem {
    name: string;
    readonly size: number;
    readonly height: number;
    readonly width: number;
    readonly data: string; // dataURL format: RFC 2397
    blobs: string;

    constructor(name: string, size: number, height: number, width: number, data: string, blobs: string) {
        this.name = name;
        this.size = size;
        this.height = height;
        this.width = width;
        this.data = data;
        this.blobs = blobs;
    }
}

class ImageProject {
    static version = "1.0.0";
    name: string;
    create: Date;
    private items: Array < ImageProjectItem > ;

    // Current item
    private current_index: number;
    private current_image: HTMLImageElement;

    // Statics
    image_loading: number;
    image_load_ok: number;
    image_load_err: number;

    constructor(name: string) {
        this.name = name;
        this.create = new Date();
        this.items = new Array < ImageProjectItem > ();

        this.current_index = -1;
        this.current_image = new Image() as HTMLImageElement;

        this.image_loading = 0;
        this.image_load_ok = 0;
        this.image_load_err = 0;
    }

    count(): number {
        return this.items.length;
    }

    empty(): boolean {
        return this.items.length < 1;
    }

    // ONLY Current Write Interface
    go(index: number): boolean {
        if (index < 0 || index >= this.items.length)
            return false;
        if (this.current_index != index) {
            this.current_index = index;
            this.current_image.src = this.items[index].data;
        }
        return true;
    }

    // ONLY Current Read Interface
    current(): [HTMLImageElement, number] {
        return [this.current_image, this.current_index];
    }

    goFirst(): boolean {
        return this.go(0);
    }

    goPrev(): boolean {
        return this.go(this.current_index - 1);
    }

    goNext(): boolean {
        return this.go(this.current_index + 1);
    }

    goLast(): boolean {
        return this.go(this.items.length - 1);
    }

    load(file: File) {
        this.image_loading++;
        loadDataURLFromFile(file).then((url: string) => {
                // dataURL ok ?
                dataURLToImage(url).then((img: HTMLImageElement) => {
                        let unit = new ImageProjectItem(file.name, file.size, img.height, img.width, url, "");
                        this.items.push(unit);
                        // Goto first ?
                        if (this.current_index < 0 || this.current_index >= this.items.length)
                            this.go(0);
                        this.image_load_ok++;
                        this.image_loading--;
                    })
                    .catch((error) => {
                        this.image_load_err++;
                        this.image_loading--;
                    });
                // Decode end
            })
            .catch((error) => {
                this.image_load_err++;
                this.image_loading--;
            }); // loadDataURLFromFile end
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
        return "Project " + this.name +
            ", version: " + ImageProject.version +
            ", create time: " + this.create +
            ", load: " + this.image_load_ok + " ok" +
            ", " + this.image_load_err + " error" +
            ", " + this.image_loading + " going.";
    }
}

enum ImageOpcode {
    Clean,
    Zoom,
    Color,
    Patch
};

class AbHead {
    // CPU: 0x12345678, Storage: [78,56,34,12], Big Endian
    t: string; // 2 bytes, Message type
    len: number; // 4 bytes, Message body length
    crc: number; // 2 bytes, CRC16

    constructor() {
        this.t = "ab"; // charCodeAt() ?
        this.len = 0;
        this.crc = 0;
    }

    encode(): ArrayBuffer {
        let p = new ArrayBuffer(8);
        let c = new Uint8Array(p);
        let dv = new DataView(p);

        c[0] = this.t.charCodeAt(0);
        c[1] = this.t.charCodeAt(1);
        dv.setUint32(2, this.len, true);
        dv.setUint16(6, crc16(c, 6), true);

        return p;
    }

    decode(p: ArrayBuffer): boolean {
        let c = new Uint8Array(p);
        let dv = new DataView(p);

        this.t = String.fromCharCode(c[0]) + String.fromCharCode(c[1]);
        this.len = dv.getUint32(2, true);
        this.crc = dv.getUint16(6, true);
        let crc = crc16(c, 6);

        return this.crc == crc;
    }
}

function isAbMessage(p: ArrayBuffer): boolean {
    let h = new AbHead();
    return h.decode(p) && (h.len + 8) == p.byteLength;
}

class ImageHead {
    // HxWxC
    h: number; // 2 bytes
    w: number; // 2 bytes
    c: number; // 2 bytes
    opc: number; // opcode, 2 byte

    constructor() {
        this.h = 0;
        this.w = 0;
        this.c = 4; // Channel
        this.opc = 0;
    }

    encode(): ArrayBuffer {
        let p = new ArrayBuffer(8);
        let b = new Uint16Array(p);
        b[0] = this.h & 0xffff;
        b[1] = this.w & 0xffff;
        b[2] = this.c & 0xffff;
        b[3] = this.opc & 0xffff;

        return p;
    }

    decode(p: ArrayBuffer) {
        let b = new Uint16Array(p);
        this.h = b[0];
        this.w = b[1];
        this.c = b[2];
        this.opc = b[3];
    }

    dataSize(): number {
        return this.h * this.w * this.c;
    }
}

function isImageMessage(p: ArrayBuffer): boolean {
    let h = new ImageHead();
    h.decode(p.slice(8, 16));
    return isAbMessage(p) && (h.dataSize() + 8 + 8) == p.byteLength;
}

const DEFAULT_WEBSOCKET_RECONNECT_INTERVAL = 30 * 1000; // 30s

class AbClient {
    private address: string;
    private socket: any; // WebSocket;
    private status: number;
    private timer: number; // Timer
    evhandler_registed: boolean;

    constructor(address: string) {
        this.address = address;
        this.socket = null;
        // Define status for socket.readyState could not be used(because socket == null)
        this.status = WebSocket.CLOSED;
        this.timer = 0; // Re-connect timer
        this.evhandler_registed = false;

        this.open();
    }

    open() {
        // console.log("Start re-connect timer...");
        if (this.timer <= 0) {
            this.timer = setInterval(() => {
                this.open();
            }, DEFAULT_WEBSOCKET_RECONNECT_INTERVAL);
        }

        if (this.status == WebSocket.CONNECTING || this.status == WebSocket.OPEN) {
            console.log("WebSocket is going on ...");
            return;
        }

        this.socket = new WebSocket(this.address);
        this.socket.binaryType = "arraybuffer";
        this.evhandler_registed = false;

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
            let start_time = new Date();
            if (this.status != WebSocket.OPEN) {
                reject("WebSocket not opened.");
            }

            if (! this.evhandler_registed) {
                this.evhandler_registed = true;

                this.socket.addEventListener('message', (event: MessageEvent) => {
                    if (event.data instanceof String) {
                        console.log("Received string data ... ", event.data);
                    }
                    if (event.data instanceof ArrayBuffer) {
                        console.log("Received ArrayBuffer ... ", event.data);
                        if (isAbMessage(event.data)) {
                            console.log("Now: ", new Date(), "start_time:",  start_time)
                            console.log("Spend", (new Date()).getTime() - start_time.getTime(), "ms for transform.");
                            resolve(event.data);
                        } else {
                            reject("Received data is not valid ArrayBuffer.");
                        }
                    }
                }, false);

                this.socket.addEventListener('error', (event: Event) => {
                    reject("WebSocket error.");
                }, false);
            }
            // Send all data in the list of arraybuffer
            for (let x of ablist) {
                this.socket.send(x);
            }
            start_time = new Date();
        });
    }

    sendImage(head: ImageHead, data: ArrayBuffer): Promise < ArrayBuffer > {
        let abhead = new AbHead();
        abhead.t = "image";
        abhead.len = 8 + data.byteLength; // ImageHead size is 8
        return this.send([abhead.encode(), head.encode(), data]);
    }

    clean(head: ImageHead, data: ArrayBuffer): Promise < ArrayBuffer > {
        head.opc = ImageOpcode.Clean;
        return this.sendImage(head, data);
    }

    zoom(head: ImageHead, data: ArrayBuffer): Promise < ArrayBuffer > {
        head.opc = ImageOpcode.Zoom;
        return this.sendImage(head, data);
    }

    color(head: ImageHead, data: ArrayBuffer): Promise < ArrayBuffer > {
        head.opc = ImageOpcode.Color;
        return this.sendImage(head, data);
    }

    patch(head: ImageHead, data: ArrayBuffer): Promise < ArrayBuffer > {
        head.opc = ImageOpcode.Patch;
        return this.sendImage(head, data);
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
//
// head.dataSize() == data.byteLength
// new AbClient("ws://localhost:8080").sendImage(head, data)
// .then((ab:ArrayBuffer) => {
//      isImageMessage(ab) ?
//      image_data = new Uint8Array(ab, 8 + 8);
// })
// .catch((errmsg) => {
//      console.log(errmsg);
// });