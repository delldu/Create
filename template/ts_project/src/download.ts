// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************
function download(href: string, filename: string) {
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

function saveTextAsFile(text: string, filename: string) {
    let href = URL.createObjectURL(text);
    download(href, filename);
}

function saveDataURLAsImage(dataurl: string, filename: string) {
    // mime --  MIME(Multipurpose Internet Mail Extensions)
    let mime = 'image/png';

    if (dataurl.startsWith('data:')) {
        let c1 = dataurl.indexOf(':', 0);
        let c2 = dataurl.indexOf(';', c1);
        mime = dataurl.substring(c1 + 1, c2);
    }

    dataurl.replace(mime, "image/octet-stream");
    download(dataurl, filename);
}

function saveCanvasAsImage(id: string, filename: string): boolean {
    let canvas = document.getElementById(id) as HTMLCanvasElement;
    if (!canvas) {
        console.log("saveCanvasAsImage: Canvas ", id, " maybe not exist.");
        return false;
    }

    // extract image data from canvas
    let mime = 'image/png';
    let dataurl = canvas.toDataURL(mime);
    saveDataURLAsImage(dataurl, filename);
    return true;
}
