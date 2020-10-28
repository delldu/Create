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

    // Bounding Box for two points
    points_bbox(p1: Point, p2: Point): Box {
        let box = new Box(0, 0, 0, 0);
        if (p1.x > p2.x) {
            box.x = p2.x;
            box.w = p1.x - p2.x;
        } else {
            box.x = p1.x;
            box.w = p2.x - p1.x;
        }
        if (p1.y > p2.y) {
            box.y = p2.y;
            box.h = p1.y - p2.y;
        } else {
            box.y = p1.y;
            box.h = p2.y - p1.y;
        }
        return box;
    }

    bbox(): Box {
        return this.points_bbox(this.start, this.stop);
    }

    // Bounding box for moving
    moving_bbox(): Box {
        return this.points_bbox(this.start, this.moving);
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
    stack: Array < number > ;

    constructor() {
        this.stack = new Array < number > ();
    }

    push(e: KeyboardEvent) {
        if (e.type == "keydown") {
            if (e.key == "Contrl")
                this.stack.push(KeyboardMode.CtrlKeydown);
            else if (e.key == "Shift") {
                this.stack.push(KeyboardMode.ShiftKeydown);
            } else if (e.key == "Alt") {
                this.stack.push(KeyboardMode.AltKeydown);
            }
        } else if (e.type == "keyup") {
            if (e.key == "Contrl")
                this.stack.push(KeyboardMode.CtrlKeyup);
            else if (e.key == "Shift") {
                this.stack.push(KeyboardMode.ShiftKeyup);
            } else if (e.key == "Alt") {
                this.stack.push(KeyboardMode.AltKeyup);
            }
        }
    }

    mode(): number {
        if (this.stack.length < 1)
            return KeyboardMode.Normal;
        return this.stack[this.stack.length - 1];
    }

    pop() {
        if (this.stack.length < 1)
            return;
        let m = this.stack.pop() as number;
        if (m != KeyboardMode.CtrlKeyup && m != KeyboardMode.ShiftKeyup && m != KeyboardMode.AltKeyup) {
            this.stack.push(m); // restore stack
            return;
        }
        let s = KeyboardMode.Normal;
        if (m == KeyboardMode.CtrlKeyup) {
            s = KeyboardMode.CtrlKeydown;
        } else if (m == KeyboardMode.ShiftKeyup) {
            s = KeyboardMode.ShiftKeydown;
        } else if (m == KeyboardMode.AltKeyup) {
            s = KeyboardMode.AltKeydown;
        }
        for (let i = this.stack.length - 1; i >= 0; i--) {
            if (s == this.stack[i]) {
                this.stack.splice(i, 1);    // pop relative keydown
                return;
            }
        }
    }

    reset() {
        this.stack.length = 0;
    }
}
