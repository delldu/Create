// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

enum MouseStatus {
    ClickOver,
    DragOver,
    Moving  // Default Status ?
}

// How to get mouse status out of event handlers ? Record it !
class Mouse {
    start: Point;
    moving: Point;
    stop: Point;
    left_button_pressed: boolean;   // mouse moving erase the status, we define it !

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

    status(): number {
        let d = this.start.distance(this.stop);
        if (d <= Point.THRESHOLD)
            return MouseStatus.ClickOver;
        // Dragging ?
        if (this.pressed())
            return MouseStatus.DragOver;
        // Default
        return MouseStatus.Moving;
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

// How to get key out of event handlers ? Record it !
class Keyboard {
    e: KeyboardEvent;

    constructor() {
        this.e = new KeyboardEvent("");
    }

    ctrlPressed(): boolean {
        return this.e.ctrlKey;
    }

    shiftPressed(): boolean {
        return this.e.shiftKey;
    }

    altPressed(): boolean {
        return this.e.altKey;
    }

    key(): string {
        return this.e.key; // Shift, Control, Alt, Escape, A etc.
    }

    type(): string {
        return this.e.type; // keydown, keyup
    }

    set(e: KeyboardEvent) {
        this.e = e;
    }
}
