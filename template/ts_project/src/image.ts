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
// function dataURLToImageData(url: string): Promise < [number, number, ImageData] > {
//     return new Promise(function(resolve, reject) {
//         if (url == null)
//             reject("dataURLToImageData: url == null");
//         let canvas = document.createElement('canvas'),
//             context = canvas.getContext('2d'),
//             image = new Image();
//         image.addEventListener('load', function() {
//             canvas.width = image.width;
//             canvas.height = image.height;
//             if (context) {
//                 context.drawImage(image, 0, 0, canvas.width, canvas.height);
//                 resolve([canvas.height, canvas.width, context.getImageData(0, 0, canvas.width, canvas.height)]);
//             }
//         }, false);
//         image.addEventListener('load', function() {
//             reject("dataURLToImageData: error.");
//         }, false);
//         image.src = url;
//     });
// }

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

function loadTextFromFile(file: File): Promise < string > {
    return new Promise(function(resolve, reject) {
        if (!(file instanceof File))
            reject("loadTextFromFile: input is not File object.");
        else {
            let reader = new FileReader();
            reader.addEventListener("error", () => {
                reject("loadTextFromFile: file read error.");
            }, false);
            reader.addEventListener("load", () => {
                // reader ok ?
                resolve(reader.result as string); // throw dataURL data
            }, false);
            reader.readAsText(file, 'utf-8');
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
    private readonly current_image: HTMLImageElement;

    // Statics
    image_loading: number;
    image_load_ok: number;
    image_load_err: number;

    // Need saving ?
    need_saving: boolean;

    // Refresh message
    refresh: Refresh;

    constructor(name: string) {
        this.name = name;
        this.create = new Date();
        this.items = new Array < ImageProjectItem > ();

        this.current_index = -1;
        this.current_image = new Image() as HTMLImageElement;

        this.image_loading = 0;
        this.image_load_ok = 0;
        this.image_load_err = 0;

        this.need_saving = false;

        this.refresh = Refresh.getInstance();
    }

    // count(): number {
    //     return this.items.length;
    // }

    // empty(): boolean {
    //     return this.items.length < 1;
    // }

    // ONLY Current Write Interface
    go(index: number): boolean {
        if (index < 0 || index >= this.items.length)
            return false;
        if (this.current_index != index) {
            this.current_index = index;
            // this.current_image.src = this.items[index].data;
        }
        return true;
    }

    indexOk(): boolean {
        return this.current_index >= 0 && this.current_index < this.items.length;
    }

    // ONLY Current Read Interface
    // current(): [HTMLImageElement, number] {
    //     return [this.current_image, this.current_index];
    // }

    // goFirst(): boolean {
    //     return this.go(0);
    // }

    // goPrev(): boolean {
    //     return this.go(this.current_index - 1);
    // }

    // goNext(): boolean {
    //     return this.go(this.current_index + 1);
    // }

    goLast(): boolean {
        return this.go(this.items.length - 1);
    }

    empty(): boolean {
        return this.items.length < 1;
    }

    key(): string {
        let i = this.current_index;
        if (i >= 0 && i < this.items.length)
            return this.items[i].name + "_" + this.items[i].size.toString();
        return "";
    }

    find(name: string, size: number): boolean {
        for (let i = 0; i < this.items.length; i++) {
            if (this.items[i].name == name && this.items[i].size == size)
                return true;
        }
        return false;
    }

    load(file: File) {
        // Remove duplicate file
        if (this.find(file.name, file.size))
            return;
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

                        this.need_saving = true;
                        this.refresh.notify("refresh_file_name_list");
                    })
                    .catch((error) => {
                        this.image_load_err++;
                        this.image_loading--;
                        console.log("error: ", error);
                    });
                // Decode end
            })
            .catch((error) => {
                this.image_load_err++;
                this.image_loading--;
                console.log("error:", error);
            }); // loadDataURLFromFile end
    }

    // <ul>
    //     <li onclick='jump_to_image(0)'>[1] 01_noise.png</li>
    //     <li onclick='jump_to_image(1)' class='sel'>[2] 02_noise.png</li>
    //     <li onclick='jump_to_image(2)'>[3] 03_noise.png</li>
    // </ul>
    listHtml(): string {
        let html = [];
        html.push("<ul>");
        for (let i = 0; i < this.items.length; i++) {
            let s = "<li onclick='jump_to_image(" + i + ")'";
            if (i == this.current_index)
                s += " class='sel'";
            let no = (i + 1).toString();
            while (no.length < 3)
                no = "0" + no;
            s += ">[" + no + '] ' + this.items[i].name + "</li>";
            html.push(s);
        }
        html.push("</ul>");
        return html.join("\n");
    }

    // JSON format file
    loadFromJSON(text: string) {
        try {
            let d = JSON.parse(text);
            // Reference this.save()
            // let save_project = {
            //     'name': this.name,
            //     'create': this.create,
            //     'items': this.items
            // };
            if (d['name'])
                this.name = d['name'];
            if (d['create'])
                this.create = new Date(Date.parse(d['create']));
            if (d['items']) {
                // this.items = ...
                this.items.length = 0; // reset() = new Array < ImageProjectItem > ();
                for (let i in d['items']) {
                    if (!d['items'].hasOwnProperty(i))
                        continue;
                    let x = d['items'][i];
                    if (!x.hasOwnProperty('name') || !x.hasOwnProperty('height') ||
                        !x.hasOwnProperty('width') || !x.hasOwnProperty('data') || !x.hasOwnProperty('blobs'))
                        continue;
                    let unit = new ImageProjectItem(x.name,
                        parseInt(x.size), parseInt(x.height), parseInt(x.width), x.data, x.blobs);
                    this.items.push(unit);
                }
            }
            this.need_saving = false;
            this.refresh.notify("refresh_file_name_list");
            this.go(0);
        } catch {
            console.log("ImageProject loadFromJSON: error");
            return;
        }
    }

    // Load model from JSON file
    open() {
        this.need_saving = false;
        // mime -- MIME(Multipurpose Internet Mail Extensions)
        let input = document.createElement('input') as HTMLInputElement;
        input.type = 'file';
        input.accept = '.json'; // mime -- '.json'; JSON File
        input.multiple = false;

        input.addEventListener('change', () => {
            if (input.files != undefined) {
                let file = input.files[0];
                loadTextFromFile(file).then((text) => {
                        this.loadFromJSON(text);
                        this.refresh.notify("refresh_project_name");
                    })
                    .catch((error) => {
                        console.log("ImageProject open: file reading error.", error);
                    });
            } else {
                console.log("ImageProject open: error.");
            }
        }, false);
        input.click();
    }

    // Save model as JSON file
    save() {
        let filename = this.name + ".json";
        let save_project = {
            'name': this.name,
            'create': this.create,
            'version': ImageProject.version,
            'items': this.items
        };
        saveTextAsFile(JSON.stringify(save_project, undefined, 2), filename);
        this.need_saving = false;
    }

    addFiles() {
        // mime -- MIME(Multipurpose Internet Mail Extensions)
        let input = document.createElement('input') as HTMLInputElement;
        input.type = 'file';
        input.accept = 'image/*'; // mime -- 'image/*';
        input.multiple = true;

        input.addEventListener('change', () => {
            if (input.files != undefined) {
                for (let i = 0; i < input.files.length; i++)
                    this.load(input.files[i]);
            } else {
                console.log("ImageProject addFiles: error.");
            }
        }, false);
        // input.dispatchEvent(new MouseEvent('click'));
        input.click();
    }

    // Delete current file
    deleteFile() {
        let index = this.current_index;
        this.items.splice(index, 1); // delete index
        if (index > this.items.length - 1)
            this.goLast();
        this.need_saving = true;
        this.refresh.notify("refresh_file_name_list");
    }
}

enum ImageOpcode {
    Clean,
    Zoom,
    Color,
    Patch
}

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

    // decode(p: ArrayBuffer) {
    //     let b = new Uint16Array(p);
    //     this.h = b[0];
    //     this.w = b[1];
    //     this.c = b[2];
    //     this.opc = b[3];
    // }

    // dataSize(): number {
    //     return this.h * this.w * this.c;
    // }
}

// function isImageMessage(p: ArrayBuffer): boolean {
//     let h = new ImageHead();
//     h.decode(p.slice(8, 16));
//     return isAbMessage(p) && (h.dataSize() + 8 + 8) == p.byteLength;
// }

class AbClient {
    static WEBSOCKET_RECONNECT_INTERVAL = 30 * 1000; // 30s

    private readonly address: string;
    private socket: any; // WebSocket;
    private status: number;
    private timer: number; // Timer
    handle_registered: boolean;

    constructor(address: string) {
        this.address = address;
        this.socket = null;
        // Define status for socket.readyState could not be used(because socket == null)
        this.status = WebSocket.CLOSED;
        this.timer = 0; // Re-connect timer
        this.handle_registered = false;

        this.open();
    }

    open() {
        // console.log("Start re-connect timer...");
        if (this.timer <= 0) {
            this.timer = setInterval(() => {
                this.open();
            }, AbClient.WEBSOCKET_RECONNECT_INTERVAL);
        }

        if (this.status == WebSocket.CONNECTING || this.status == WebSocket.OPEN) {
            console.log("WebSocket is going on ...");
            return;
        }

        this.socket = new WebSocket(this.address);
        this.socket.binaryType = "arraybuffer";
        this.handle_registered = false;

        this.socket.addEventListener('open', (event: Event) => {
            console.log("WebSocket open on " + this.socket.url + " ...", event);
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
            if (!this.handle_registered) {
                this.handle_registered = true;

                this.socket.addEventListener('message', (event: MessageEvent) => {
                    if (event.data instanceof String) {
                        console.log("Received string data ... ", event.data);
                    }
                    if (event.data instanceof ArrayBuffer) {
                        console.log("Received ArrayBuffer ... ", event.data);
                        if (isAbMessage(event.data)) {
                            console.log("Spend", (new Date()).getTime() - start_time.getTime(), "ms for transform.");
                            resolve(event.data);
                        } else {
                            reject("Received data is not valid ArrayBuffer.");
                        }
                    }
                }, false);

                this.socket.addEventListener('error', (event: Event) => {
                    console.log("error: ", event);
                    reject("WebSocket error.");
                }, false);
            }
            // Send all data in the list of arraybuffer
            for (let x of ablist) {
                this.socket.send(x);
            }
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
// .catch((error) => {
//      console.log(error);
// });