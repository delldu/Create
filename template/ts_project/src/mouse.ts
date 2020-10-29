// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

enum MouseOverStatus {
    DragOver,
    ClickOver,
}

// How to get mouse overStatus out of event handlers ? Record it !
class Mouse {
    start: Point;
    moving: Point;
    stop: Point;
    left_button_pressed: boolean; // mouse moving erase the overStatus, we define it !

    constructor() {
        this.start = new Point(0, 0);
        this.moving = new Point(0, 0);
        this.stop = new Point(0, 0);

        this.left_button_pressed = false;
    }

    set(e: MouseEvent, scale: number = 1.0) {
        if (e.type == "mousedown") {
            this.start.x = e.offsetX / scale;
            this.start.y = e.offsetY / scale;
            this.left_button_pressed = true;
        } else if (e.type == "mouseup") {
            this.stop.x = e.offsetX / scale;
            this.stop.y = e.offsetY / scale;
        } else if (e.type == "mousemove") {
            this.moving.x = e.offsetX / scale;
            this.moving.y = e.offsetY / scale;
        }
    }

    reset() {
        this.left_button_pressed = false;
    }

    pressed(): boolean {
        return this.left_button_pressed;
    }

    overStatus(): number {
        let d = this.start.distance(this.stop);
        return (this.pressed() && d > Point.THRESHOLD) ?
            MouseOverStatus.DragOver : MouseOverStatus.ClickOver;
    }

    bbox(): Box {
        return Box.bbox(this.start, this.stop);
    }

    // Bounding box for moving
    moving_bbox(): Box {
        return Box.bbox(this.start, this.moving);
    }

    delta(): [number, number] {
        return [this.stop.x - this.start.x, this.stop.y - this.start.y];
    }

    moving_delta(): [number, number] {
        return [this.moving.x - this.start.x, this.moving.y - this.start.y];
    }
}

enum KeyboardMode {
    Normal,
    CtrlKeydown,
    CtrlKeyup,
    ShiftKeydown,
    ShiftKeyup,
    AltKeydown,
    AltKeyup
}

// How to get key out of event handlers ? Record it !
class Keyboard {
    stack: Array < KeyboardMode > ;

    constructor() {
        this.stack = new Array < KeyboardMode > ();
    }

    push(e: KeyboardEvent) {
        if (e.type == "keydown") {
            if (e.key == "Control")
                this.stack.push(KeyboardMode.CtrlKeydown);
            else if (e.key == "Shift") {
                this.stack.push(KeyboardMode.ShiftKeydown);
            } else if (e.key == "Alt") {
                this.stack.push(KeyboardMode.AltKeydown);
            }
        } else if (e.type == "keyup") {
            if (e.key == "Control")
                this.stack.push(KeyboardMode.CtrlKeyup);
            else if (e.key == "Shift") {
                this.stack.push(KeyboardMode.ShiftKeyup);
            } else if (e.key == "Alt") {
                this.stack.push(KeyboardMode.AltKeyup);
            }
        }
    }

    mode(): KeyboardMode {
        if (this.stack.length < 1)
            return KeyboardMode.Normal;
        return this.stack[this.stack.length - 1];
    }

    pop() {
        if (this.stack.length < 1)
            return;
        let m = this.stack.pop() as KeyboardMode;
        if (m != KeyboardMode.CtrlKeyup && m != KeyboardMode.ShiftKeyup && m != KeyboardMode.AltKeyup) {
            this.stack.push(m); // Restore stack
            return;
        }
        let s = KeyboardMode.Normal; // Search 
        switch (m) {
            case KeyboardMode.CtrlKeyup:
                s = KeyboardMode.CtrlKeydown;
                break;
            case KeyboardMode.ShiftKeyup:
                s = KeyboardMode.ShiftKeydown;
                break;
            case KeyboardMode.AltKeyup:
                s = KeyboardMode.AltKeydown;
                break;
            default:
                break;
        }
        for (let i = this.stack.length - 1; i >= 0; i--) {
            if (s == this.stack[i]) {
                this.stack.splice(i, 1); // pop relative keydown
                break;
            }
        }
    }

    reset() {
        this.stack.length = 0;
    }
}