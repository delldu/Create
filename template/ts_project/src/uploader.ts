// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

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
