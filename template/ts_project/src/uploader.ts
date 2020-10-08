// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

function loadXMLDoc() {
    let xmlhttp = new XMLHttpRequest();

    xmlhttp.onreadystatechange = function() {
        if (xmlhttp.readyState == 4 && xmlhttp.status == 200) {
            // document.getElementById("myDiv").innerHTML = xmlhttp.responseText;
        }
    }
    xmlhttp.open("POST", "/ajax/demo_post2.asp", true);
    xmlhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xmlhttp.send("fname=Bill&lname=Gates");
}

function createInputFile(): HTMLInputElement {
    let input = document.createElement('input') as HTMLInputElement;
    input.type = 'file';
    input.accept = 'image/*';
    input.multiple = true;

    return input;
}

function selectFiles() {
    return new Promise(function(resolve: (value: FileList | null) => void, reject) {
        let input = createInputFile();
        input.addEventListener('change', () => {
            resolve(input.files || null);
        }, false);

        setTimeout(() => {
            let event = new MouseEvent('click');
            input.dispatchEvent(event);
        }, 10); // 10 ms
    });
}

class uploadFile {
    xhr: XMLHttpRequest;

    constructor() {
        this.xhr = new XMLHttpRequest();
    }

    public start(file: File) {
        let url = "http://upload.lwio.me";

        let form = new FormData();
        form.append("file", file);

        this.xhr.open("post", url, true); // true is async

        this.xhr.addEventListener("progress", (event: ProgressEvent) => {
            // ProgressEvent.loaded ， ProgressEvent.total
            console.log(event.loaded);
        }, false);

        this.xhr.addEventListener("load", (event: Event) => {
            console.log(this.xhr.responseText);
        }, false);

        this.xhr.addEventListener("error", (event: Event) => {
            console.log("Upload failed.")
        }, false);

        this.xhr.addEventListener("abort", (event: Event) => {
            console.log("Abort.");
        }, false);

        this.xhr.send(form);
    }

    public cancle() {
        this.xhr.abort();
    }
}

// <body>
//        <progress id="progressBar" value="0" max="100" style="width: 300px;"></progress>
//        <span id="percentage"></span>
//        <span id="time"></span>
//        <br><br>
//        <input type="file" id="files" class="upload-input" onchange="uploadFile.prototype.showUploadInfo(this.files)" multiple />
//        <div class="upload-button" onclick="uploadFile.prototype.postFile()">upload</div>
//        <div class="upload-button" onclick="uploadFile.prototype.cancleUploadFile()">cancel</div>
//        <output id="list"></output>
//    </body>
//