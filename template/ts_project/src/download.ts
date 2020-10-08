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

    // Simulate a mouse click event
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