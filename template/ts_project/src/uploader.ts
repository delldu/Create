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

class AJax {
    xhr: XMLHttpRequest;

    constructor() {
        this.xhr = new XMLHttpRequest();
    }

    // post, get, ...
    // text/binary/image ... ?

    get(url: string) {

    }

    post(url: string, file: File) {
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

    cancle() {
        this.xhr.abort();
    }
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