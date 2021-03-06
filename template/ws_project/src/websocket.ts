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
const dataURLToImage = (url: string) => {
    let e = new Image() as HTMLImageElement;
    return new Promise(function(resolve: (value: HTMLImageElement) => void, reject) {
        // Promise excutor ...
        if (url == null) {
            reject();
        }
        e.addEventListener('load', function() {
            resolve(e);
        }, false);
        e.addEventListener('abort', function() {
            reject();
        }, false);
        e.addEventListener('error', function() {
            reject();
        }, false);
        e.src = url;
    });
}

// Convert dataURL to ImageData (ArrayBuffer)
// This is useful for NImage ...
const dateURLToImageData = (url: string) => {
    return new Promise(function(resolve, reject) {
        if (url == null)
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
        image.src = url;
    });
}

// dateURLToImageData(URI).then((imageData) => {
//     // Here you can use ImageData
//     console.log(imageData);
// });

// Load dataURL from file
const loadDataURLFromFile = (file: File) => {
    return new Promise(function(resolve: (value: string) => void, reject) {
        if (!(file instanceof File))
            reject();
        else {
            let reader = new FileReader();
            reader.addEventListener("error", () => {
                reject();
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

            Download(url, "a.png");

            dataURLToImage(url).then((img:HTMLImageElement) => {
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

class ShiftPanel {
    panels: Array < string > ;

    constructor() {
        this.panels = new Array < string > ();
    }

    add(id: string) {
        if (this.panels.findIndex(e => e == id) >= 0) {
            console.log("ShiftPanel: Duplicate id", id);
            return;
        }
        let element = document.getElementById(id);
        if (!element) {
            console.log("ShiftPanel: Invalid element id ", id);
            return;
        }
        this.panels.push(id);
    }

    click(id: string) {
        for (let i = 0; i < this.panels.length; i++) {
            // add make sure valid element, but compiler donot know !
            let element = document.getElementById(this.panels[i]);
            if (element)
                element.style.display = (id == this.panels[i]) ? "" : "none";
        }
    }

    clicked(): number {
        for (let i = 0; i < this.panels.length; i++) {
            // add make sure valid element, but compiler donot know !
            let element = document.getElementById(this.panels[i]);
            if (element && element.style.display != "none")
                return i;
        }
        return -1;
    }

    shift() {
        if (this.panels.length < 1)
            return;

        let index = this.clicked();
        index = (index < 0) ? 0 : index + 1;
        if (index > this.panels.length - 1)
            index = 0;
        for (let i = 0; i < this.panels.length; i++) {
            // add make sure valid element, but compiler donot know !
            let element = document.getElementById(this.panels[i]);
            if (element)
                element.style.display = (i == index) ? "" : "none";
        }
    }

    test() {
        // define switch method
        console.log("ShiftPanel: press shift key for test.");
        window.addEventListener("keydown", (e: KeyboardEvent) => {
            if (e.key == 'Shift')
                this.shift();
            e.preventDefault();
        }, false);
    }
}

enum NImageOpcode {
    Clean,
    Zoom,
    Color,
    Patch
};

class NImageHead {
    // CxHxW
    c: number; // 2 bytes, general == 4 for RGBA
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

        return p;
    }

    decode(p: ArrayBuffer): boolean {
        let b = new Uint16Array(p);
        let c = new Uint8Array(p);

        this.c = b[0];
        this.h = b[1];
        this.w = b[2];
        this.opcode = c[6];

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
                crc = (odd) ? (crc ^ 0x07 % 256) : (crc % 256);
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

// NImagePerformance
function NImagePerformance() {
    let start_time = (new Date()).getTime();
    for (let i = 0; i < 1000; i++) {
        let x = new NImage();
        x.head = new NImageHead();
        x.head.setSize(4, 2048, 4096);  // 8K Image
        // h.setSize(4, 1024, 2048);
        x.head.opcode = NImageOpcode.Patch;
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
// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************
function Download(href: string, filename: string) {
    let a = document.createElement('a');
    a.href = href;
    a.target = '_blank';
    a.download = filename;

    // simulate a mouse click event
    var event = new MouseEvent('click', {
        view: window,
        bubbles: true,
        cancelable: true
    });

    a.dispatchEvent(event);
}

function SaveTextAsFile(text: string, filename: string) {
    let href = URL.createObjectURL(text);
    Download(href, filename);
}


function SaveDataURLAsImage(dataurl: string, filename: string) {
    let img_mime = 'image/png';

    if (dataurl.startsWith('data:')) {
        let c1 = dataurl.indexOf(':', 0);
        let c2 = dataurl.indexOf(';', c1);
        img_mime = dataurl.substring(c1 + 1, c2);
    }

    dataurl.replace(img_mime, "image/octet-stream");
    Download(dataurl, filename);
}

function SaveCanvasAsImage(id: string, filename: string) {
    let canvas = document.getElementById(id) as HTMLCanvasElement;
    if (!canvas) {
        console.log("Canvas not exist? id is ", id);
        return;
    }

    // extract image data from canvas
    let img_mime = 'image/png';
    let dataurl = canvas.toDataURL(img_mime);
    SaveDataURLAsImage(dataurl, filename);
}