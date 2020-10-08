// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

class ShiftPanel {
    panels: Array < string > ;

    constructor() {
        this.panels = new Array < string > ();
    }

    add(id: string) {
        if (this.panels.findIndex(e => e == id) >= 0) {
            console.log("ShiftPanel: Duplicate id", id);
            return;
        }
        let element = document.getElementById(id);
        if (!element) {
            console.log("ShiftPanel: Invalid element id ", id);
            return;
        }
        this.panels.push(id);
    }

    click(id: string) {
        for (let i = 0; i < this.panels.length; i++) {
            // add make sure valid element, but compiler donot know !
            let element = document.getElementById(this.panels[i]);
            if (element)
                element.style.display = (id == this.panels[i]) ? "" : "none";
        }
    }

    clicked(): number {
        for (let i = 0; i < this.panels.length; i++) {
            // add make sure valid element, but compiler donot know !
            let element = document.getElementById(this.panels[i]);
            if (element && element.style.display != "none")
                return i;
        }
        return -1;
    }

    shift() {
        if (this.panels.length < 1)
            return;

        let index = this.clicked();
        index = (index < 0) ? 0 : index + 1;
        if (index > this.panels.length - 1)
            index = 0;
        for (let i = 0; i < this.panels.length; i++) {
            // add make sure valid element, but compiler donot know !
            let element = document.getElementById(this.panels[i]);
            if (element)
                element.style.display = (i == index) ? "" : "none";
        }
    }

    test() {
        // define switch method
        console.log("ShiftPanel: press shift key for test.");
        window.addEventListener("keydown", (e: KeyboardEvent) => {
            if (e.key == 'Shift')
                this.shift();
            e.preventDefault();
        }, false);
    }
}
