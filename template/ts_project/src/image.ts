
// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

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
function dateURLToImageData(url: string): Promise < ImageData > {
    return new Promise(function(resolve, reject) {
        if (url == null)
            reject("dateURLToImageData: url == null");
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
        image.src = url;
    });
}

// dateURLToImageData(URI).then((imageData) => {
//     // Here you can use ImageData
//     console.log(imageData);
// });

// Load dataURL from file
function loadDataURLFromFile(file: File): Promise<string> {
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
                // Goto first item
                if (this.current_index < 0 || this.current_index >= this.items.length)
                    this.go(0);
                this.image_load_ok++;
                this.image_loading--;
            }, () => {
                this.image_load_err++;
                this.image_loading--;
            });
            // Decode end
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

enum NImageOpcode {
    Clean,
    Zoom,
    Color,
    Patch
};

class NImageHead {
    // 4xHxW
    h: number; // 2 bytes
    w: number; // 2 bytes
    opc: number; // opcode, 2 byte
    crc: number;

    constructor() {
        this.h = 0;
        this.w = 0;
        this.opc = 0;
        this.crc = 0;
    }

    setSize(h: number, w: number) {
        this.h = (h & 0xffff);
        this.w = (w & 0xffff);
    }

    encode(): ArrayBuffer {
        let p = new ArrayBuffer(8);
        let c = new Uint8Array(p);
        let b = new Uint16Array(p);
        b[0] = this.h & 0xffff;
        b[1] = this.w & 0xffff;
        b[2] = this.opc & 0xffff;
        b[3] = this.crc16(c, 6);

        return p;
    }

    decode(p: ArrayBuffer): boolean {
        let b = new Uint16Array(p);
        let c = new Uint8Array(p);

        this.h = b[0];
        this.w = b[1];
        this.opc = b[2];
        this.crc = b[3];

        console.log("Decode ........................");
        console.log("c === ", c);
        console.log("this.crc = ", this.crc);
        console.log("this.crc16() = ", this.crc16(c, 6));

        return (this.crc == this.crc16(c, 6));
    }

    dataSize(): number {
        return 4 * this.h * this.w;
    }

    crc16(b: Uint8Array, n: number): number {
        let crc = 0;
        let CRC_CCITT_POLY = 0x1021;
        for (let i = 0; i < n; i++) {
            crc = crc ^ (b[i] << 8);
            for (let j = 0; j < 8; j++) {
                if (crc & 0x8000)
                    crc = (crc << 1)^CRC_CCITT_POLY;
                else
                    crc = crc << 1;
            }
        }
        return crc & 0xffff;
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
        console.log("head ---------", h);

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

// NImagePerformance
function NImagePerformance() {
    let start_time = (new Date()).getTime();
    for (let i = 0; i < 1000; i++) {
        let x = new NImage();
        x.head = new NImageHead();
        x.head.setSize(2048, 4096); // 8K Image
        // h.setSize(4, 1024, 2048);
        x.head.opc = NImageOpcode.Patch;
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

class NImageClient {
    wsurl: string;
    socket: WebSocket;

    constructor(wsurl: string) {
        this.wsurl = wsurl;
        this.socket = new WebSocket(wsurl);
        this.socket.binaryType = "arraybuffer";

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
            console.log("WebSocket closed for " + event.reason + "(" + event.code + ")");
        }, false);

        this.socket.binaryType = "arraybuffer";
    }

    private echo_start(x: NImage):Promise<ArrayBuffer> {
        return new Promise((resolve, reject) => {
            if (!x.valid()) {
                reject("NImageClient: Invalid input tensor.");
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
                reject("NImageClient: WebSocket open error.")
                // handle error event
            }, false);

            if (this.socket.readyState != WebSocket.OPEN) {
                reject("NImageClient: WebSocket not opened.");
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
                console.log("Received from server ", buffer.byteLength + " bytes.");
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
        x.head.opc = NImageOpcode.Clean;
        return this.echoService(x);
    }

    zoom(x: NImage): [boolean, NImage] {
        x.head.opc = NImageOpcode.Zoom;
        return this.echoService(x);
    }

    color(x: NImage): [boolean, NImage] {
        x.head.opc = NImageOpcode.Color;
        return this.echoService(x);
    }

    patch(x: NImage): [boolean, NImage] {
        x.head.opc = NImageOpcode.Patch;
        return this.echoService(x);
    }

    close() {
        this.socket.close();
    }
}

// let client = new NImageClient("socket://localhost:8080");
// let x = new NImage();
// x.head = new NImageHead();
// x.head.setSize(2048, 4096); // 8K Image
// // h.setSize(4, 1024, 2048);
// x.head.opc = NImageOpcode.Patch;
// x.data = new Uint8ClampedArray(new ArrayBuffer(x.head.dataSize()));
// let [ok, y] = client.color(x);

// console.log("OK: ", ok, y.h, y.w, y.opc, y.crc.toString(16));

