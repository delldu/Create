// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

// var kb = require("keyboard.js");
// const kb = require('keyboard.ts');
// import Keyboard from "src/keyboard.ts";

class Mouse {
    start: Point;
    moving: Point;
    stop: Point;

    pressed: boolean; // mouse pressed

    constructor() {
        this.start = new Point(0, 0);
        this.moving = new Point(0, 0);
        this.stop = new Point(0, 0);

        this.pressed = false; // mouse clicked
    }

    clone() {
        let m = new Mouse();
        m.start = this.start.clone();
        m.moving = this.moving.clone();
        m.stop = this.stop.clone();

        m.pressed = this.pressed;
        return m;
    }

    offset(deltaX: number, deltaY: number) {
        this.start.offset(deltaX, deltaY);
        this.moving.offset(deltaX, deltaY);
        this.stop.offset(deltaX, deltaY);
    }

    zoom(s: number) {
        this.start.zoom(s);
        this.moving.zoom(s);
        this.stop.zoom(s);
    }

    isclick(): boolean {
        let d = this.start.distance(this.stop);
        return d <= MOUSE_DISTANCE_THRESHOLD;
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
    mbbox(): Box {
        return this.points_bbox(this.start, this.moving);
    }
}
