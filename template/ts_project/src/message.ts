// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

"use strict";

class Message {
    timer: number;
    element: HTMLElement; // message element

    constructor(id: string) {
        this.timer = 0;
        this.element = document.getElementById(id);
        this.element.addEventListener('dblclick', function() {
            this.style.display = 'none';
        }, false);
    }

    show(msg: string, t: number) {
        if (this.timer) {
            clearTimeout(this.timer);
        }
        this.element.innerHTML = msg;
        this.timer = setTimeout(() => {
            this.element.style.display = 'none';
        }, t);
    }
}

// Test:
// <div id="message_id" class="message">Message Bar</div>
// function load() {
//     msgbar = new Message("message_id");
//     msgbar.show("This is a message ........................", 10000); // 10 s
// }