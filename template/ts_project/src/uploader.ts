// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

function loadXMLDoc() {
    let xhr = new XMLHttpRequest();

    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // document.getElementById("myDiv").innerHTML = xhr.responseText;
        }
    }
    xhr.open("POST", "/ajax/demo_post2.asp", true);
    xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhr.send("fname=Bill&lname=Gates");
}

// mime -- MIME(Multipurpose Internet Mail Extensions)
function selectFiles(mime: string): Promise < FileList > {
    return new Promise((resolve, reject) => {
        let input = document.createElement('input') as HTMLInputElement;
        input.type = 'file';
        input.accept = mime; // 'image/*';
        input.multiple = true;

        input.addEventListener('change', () => {
            if (input.files != undefined)
                resolve(input.files);
            else
                reject("selectFiles: NO file selected.");
        }, false);

        setTimeout(() => {
            let event = new MouseEvent('click');
            input.dispatchEvent(event);
        }, 10); // 10 ms
    });
}

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
// 	let data = new FormData();
// 	data.append("username", "Bill");
// 	data.append("age", 60);
function ajaxPostFormData(url: string, data: FormData): Promise < string > {
    return new Promise(function(resolve, reject) {
        let xhr = new XMLHttpRequest();
        // xhr.timeout = 10 * 1000;
    	
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
    for (let i = 0; i < files.length; i++)
        formData.append('files[]', files[i]);
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