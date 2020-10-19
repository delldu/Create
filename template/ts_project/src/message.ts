// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

class Message {
    timer: number;
    element: HTMLElement; // message element

    constructor(id: string) {
        this.timer = 0;
        this.element = document.getElementById(id) as HTMLElement;
        this.element.addEventListener('dblclick', function() {
            this.style.display = 'none';
        }, false);

        console.log("Message: double click hidden messgae.");
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

// Test:
// <div id="message_id" class="message">Message Bar</div>
// function load() {
//     msgbar = new Message("message_id");
//     msgbar.show("This is a message ........................", 10); // 10 s
// }

