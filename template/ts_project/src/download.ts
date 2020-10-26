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
    a.click();
}

function saveTextAsFile(text: string, filename: string) {
    let blob = new Blob([text], { type: 'text/json;charset=utf-8' });
    let href = URL.createObjectURL(blob);
    download(href, filename);
    URL.revokeObjectURL(href);
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

// class Refresh {
//     static THRESHOLD = 20;
//     static instance = new Refresh();
//     refresh_events: Array < string > ;
//
//     private constructor() {
//         this.refresh_events = new Array < string > ();
//     }
//
//     notify(msg: string) {
//         if (this.refresh_events.length >= 1024) {
//             this.refresh_events.shift();
//         }
//         this.refresh_events.push(msg);
//     }
//
//     message(): string {
//         console.log(Refresh.THRESHOLD);
//
//         let ok = this.refresh_events.length > 0;
//         if (ok) {
//             let msg = this.refresh_events.shift() as string;
//             let count = 0;
//             for (let i = 0; i < this.refresh_events.length; i++) {
//                 if (this.refresh_events[i] != msg) {
//                     break;
//                 } else {
//                     count++;
//                 }
//             }
//             if (count > 0) {
//                 this.refresh_events.splice(0, count);
//             }
//             return msg;
//         }
//         return "";
//     }
//
//     static getInstance(): Refresh {
//         return Refresh.instance;
//     }
// }

// let refresh = Refresh.getInstance();


// refresh.notify("abcdef");
// refresh.notify("abcdef");
// refresh.notify("1234557");
// refresh.notify("1234557");
// refresh.notify("1234557");
//
// while (true) {
//     let msg = refresh.message();
//     if (msg.length > 0) {
//         console.log(msg);
//     } else {
//         break;
//     }
// }
