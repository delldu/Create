// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************
const sleep = (time: number) => {
    return new Promise(resolve => setTimeout(resolve, time));
}
// sleep(2000).then(() => {
//     console.log("2 seconds passed.");
// })

class Message {
    timer: number;
    element: HTMLElement; // message element

    constructor(id: string) {
        this.timer = 0;
        this.element = document.getElementById(id) as HTMLElement;
        this.element.addEventListener('dblclick', function() {
            this.style.display = 'none';
        }, false);

        console.log("Message: double click hidden message.");
    }

    // show message for t seconds
    show(msg: string, t: number) {
        if (this.timer) {
            clearTimeout(this.timer);
        }
        this.element.innerHTML = msg;
        this.element.style.display = ""; // Show
        this.timer = setTimeout(() => {
            this.element.style.display = 'none';
        }, t * 1000);
    }
}

class Progress {
    element: HTMLElement;

    constructor(i: number) {
        this.element = document.getElementsByTagName("progress")[i];
    }

    show(yes: boolean) {
        this.element.style.display = (yes) ? "" : "none";
    }

    update(v: number) {
        this.element.setAttribute('value', v.toString());
        if (this.element.style.display == "none")
            this.element.style.display = "";
    }

    startDemo(value: number) {
        value = value + 1;
        this.update(value);
        if (value < 100) {
            setTimeout(() => { this.startDemo(value); }, 100);
        } else {
            this.show(false);
        }
    }
}

// Test:
// <progress value="0" max="100">Progress Bar</div>
// function load() {
//     pb = new Progress(0);
//     pb.startDemo(0); // start 0
// }

// Test:
// <div id="message_id" class="message">Message Bar</div>
// function load() {
//     bar = new Message("message_id");
//     bar.show("This is a message ........................", 10); // 10 s
// }


function download(href: string, filename: string) {
    let a = document.createElement('a');
    a.href = href;
    a.target = '_blank';
    a.download = filename;
    a.click();
}

function saveTextAsFile(text: string, filename: string) {
    let blob = new Blob([text], { type: 'text/json;charset=utf-8' });
    let href = URL.createObjectURL(blob);
    download(href, filename);
    URL.revokeObjectURL(href);
}

function saveDataURLAsImage(dataURL: string, filename: string) {
    // mime --  MIME(Multipurpose Internet Mail Extensions)
    let mime = 'image/png';

    if (dataURL.startsWith('data:')) {
        let c1 = dataURL.indexOf(':', 0);
        let c2 = dataURL.indexOf(';', c1);
        mime = dataURL.substring(c1 + 1, c2);
    }

    dataURL.replace(mime, "image/octet-stream");
    download(dataURL, filename);
}

function saveCanvasAsImage(id: string, filename: string): boolean {
    let canvas = document.getElementById(id) as HTMLCanvasElement;
    if (!canvas) {
        console.log("saveCanvasAsImage: Canvas ", id, " maybe not exist.");
        return false;
    }

    // extract image data from canvas
    let mime = 'image/png';
    let dataURL = canvas.toDataURL(mime);
    saveDataURLAsImage(dataURL, filename);
    return true;
}

class Refresh {
    static THRESHOLD = 2048;
    static instance = new Refresh();
    private readonly event_queue: Array < string > ;

    private constructor() {
        this.event_queue = new Array < string > ();
    }

    notify(message: string) {
        if (this.event_queue.length >= Refresh.THRESHOLD) {
            this.event_queue.shift();
        }
        this.event_queue.push(message);
    }

    message(prefix: string): string {
        let ok = this.event_queue.length > 0;
        if (ok) {
            let start = -1;
            let message = "";
            for (let i = 0; i < this.event_queue.length; i++) {
                if (this.event_queue[i].startsWith(prefix)) {
                    start = i;
                    message = this.event_queue[i];
                    break;
                }
            }
            if (start >= 0) { // find same message ...
                let count = 1;
                for (let i = start + 1; i < this.event_queue.length; i++) {
                    if (this.event_queue[i] == message)
                        count++;
                    else
                        break;
                }
                this.event_queue.splice(start, count);
                return message;
            }
        }
        return "";
    }

    static getInstance(): Refresh {
        return Refresh.instance;
    }
}

// let refresh = Refresh.getInstance();

// AJAX = Asynchronous JavaScript and XML
function ajaxGet(url: string): Promise < string > {
    return new Promise(function(resolve, reject) {
        let xhr = new XMLHttpRequest();
        // xhr.timeout = 10 * 1000;

        xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");

        xhr.open("GET", url, true); // true is async
        xhr.addEventListener("progress", (event: ProgressEvent) => {
            console.log("ajaxGet: loaded", event.loaded, "bytes, total", event.total, "bytes.");
        }, false);
        xhr.addEventListener("load", (event: Event) => {
            resolve(xhr.responseText); // xhr.statusText == 200 OK ?
        }, false);
        xhr.addEventListener("error", (event: Event) => {
            reject("ajaxGet: error.");
        }, false);
        xhr.addEventListener("abort", (event: Event) => {
            reject("ajaxGet: abort."); // timeout etc ...
        }, false);
        xhr.send();
    });
}

// Post ArrayBuffer, receive ArrayBuffer ...
function ajaxPostArrayBuffer(url: string, data: ArrayBuffer): Promise < ArrayBuffer > {
    return new Promise(function(resolve, reject) {
        let xhr = new XMLHttpRequest();
        // xhr.timeout = 10 * 1000;

        xhr.responseType = "arraybuffer";
        xhr.open("POST", url, true); // true is async
        xhr.addEventListener("progress", (event: ProgressEvent) => {
            console.log("ajaxPostArrayBuffer: loaded", event.loaded, "bytes, total", event.total, "bytes.");
        }, false);
        xhr.addEventListener("load", (event: Event) => {
            resolve(xhr.response as ArrayBuffer); // xhr.statusText == 200 OK ?
        }, false);
        xhr.addEventListener("error", (event: Event) => {
            reject("ajaxPostArrayBuffer: error.");
        }, false);
        xhr.addEventListener("abort", (event: Event) => {
            reject("ajaxPostArrayBuffer: abort."); // timeout etc ...
        }, false);
        xhr.send(data);
    });
}

// Create form data like ajaxPostFiles:
//  let data = new FormData();
//  data.append("username", "Bill");
//  data.append("age", 60);
function ajaxPostFormData(url: string, data: FormData): Promise < string > {
    return new Promise(function(resolve, reject) {
        let xhr = new XMLHttpRequest();
        // xhr.timeout = 10 * 1000;
        // xhr.setRequestHeader("Content-Type","application/x-www-form-urlencoded");

        xhr.open("POST", url, true); // true is async
        xhr.addEventListener("progress", (event: ProgressEvent) => {
            console.log("ajaxPostFormData: loaded", event.loaded, "bytes, total", event.total, "bytes.");
        }, false);
        xhr.addEventListener("load", (event: Event) => {
            resolve(xhr.responseText); // xhr.statusText == 200 OK ?
        }, false);
        xhr.addEventListener("error", (event: Event) => {
            reject("ajaxPostFormData: error.");
        }, false);
        xhr.addEventListener("abort", (event: Event) => {
            reject("ajaxPostFormData: abort."); // timeout etc ...
        }, false);
        xhr.send(data);
    });
}

function ajaxPostFiles(url: string, files: FileList): Promise < string > {
    let formData = new FormData();
    // formData.append(name, value, filename);
    for (let i = 0; i < files.length; i++)
        formData.append('file[]', files[i]);
    console.log("File Form Data -------- ", formData);
    return ajaxPostFormData(url, formData);
}

// <body>
//        <progress id="progressBar" value="0" max="100" style="width: 300px;"></progress>
//        <span id="percentage"></span>
//        <span id="time"></span>
//        <br><br>
//        <input type="file" id="files" class="upload-input" onchange="AJax.prototype.showUploadInfo(this.files)" multiple />
//        <div class="upload-button" onclick="AJax.prototype.postFile()">upload</div>
//        <div class="upload-button" onclick="AJax.prototype.cancleUploadFile()">cancel</div>
//        <output id="list"></output>
//    </body>
//

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
            // add make sure valid element, but compiler do not know !
            let element = document.getElementById(this.panels[i]);
            if (element)
                element.style.display = (id == this.panels[i]) ? "" : "none";
        }
    }

    clicked(): number {
        for (let i = 0; i < this.panels.length; i++) {
            // add make sure valid element, but compiler do not know !
            let element = document.getElementById(this.panels[i]);
            if (element && element.style.display != "none")
                return i;
        }
        return -1;
    }

    id(): string {
        let index = this.clicked();
        return (index >= 0) ? this.panels[index] : "";
    }

    shift() {
        if (this.panels.length < 1)
            return;

        let index = this.clicked();
        index = (index < 0) ? 0 : index + 1;
        if (index > this.panels.length - 1)
            index = 0;
        for (let i = 0; i < this.panels.length; i++) {
            // add make sure valid element, but compiler do not know !
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

