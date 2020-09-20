// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

"use strict";

class Progress {
    element: HTMLElement;

    constructor(i:number) {
        this.element = document.getElementsByTagName("progress")[0];
    }

    update(v: number) {
        this.element.setAttribute('value', v.toString());
    }

    startDemo(value:number) {
        value = value + 1;
        this.update(value);
        if (value < 100) {
            setTimeout(()=>{this.startDemo(value);}, 50);
        }
    }
}

// Test:
// <progress value="0" max="100">Progress Bar</div>
// function load() {
//     pb = new Progress(0);
//     pb.startDemo(0); // start 0
// }
