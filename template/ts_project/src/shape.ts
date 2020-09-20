// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

"use strict";

// Corner point IS control point for 2d shape objects
const CORNER_POINT_RADIUS = 3;
const CORNER_POINT_COLOR = "0xff0000";

const enum ShapeID {
    Rectangle,
    Ellipse,
    Polygon,
    Polyline
}

class Point {
    x: number;
    y: number;
    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }
    // manhatattan disance
    manhat(p: Point): number {
        return Math.max(Math.abs(p.x - this.x), Math.abs(p.y - this.y));
    }
}

abstract class Shape2d {
    id: ShapeID;

    constructor(id: ShapeID) {
        this.id = id;
    }

    abstract inside(p: Point): boolean; // p inside 2d shape object
    abstract oncorner(p: Point): boolean; // p is on corner of 2d shape object
    abstract draw(brush: any);

    drawCorner(brush: any, cx: number, cy: number) {
        brush.beginPath();
        brush.arc(cx, cy, CORNER_POINT_RADIUS, 0, 2 * Math.PI, false);
        brush.closePath();
        brush.fillStyle = CORNER_POINT_COLOR;
        brush.globalAlpha = 1.0;
        brush.fill();
    }
}

class Rectangle extends Shape2d {
    p1: Point;
    p2: Point;

    constructor(p1: Point, p2: Point) {
        super(ShapeID.Rectangle);
        this.p1 = p1;
        this.p2 = p2;

        this.normal();
    }

    normal() {
        let t;
        if (this.p1.x > this.p2.x) {
            t = this.p1.x;
            this.p1.x = this.p2.x;
            this.p2.x = t;
        }
        if (this.p1.y > this.p2.y) {
            t = this.p1.y;
            this.p1.y = this.p2.y;
            this.p2.y = t;
        }
    }

    draw(brush: any) {
        console.log("Shape ID:", this.id, ", ", this.p1, this.p2);
        // brush.beginPath();
        // brush.moveTo(this.p1.x, this.p1.y);
        // brush.lineTo(this.p2.x, this.p1.y);
        // brush.lineTo(this.p2.x, this.p2.y);
        // brush.lineTo(this.p1.x, this.p2.y);
        // brush.closePath();
    }

    inside(p: Point): boolean {
        return (p.x > this.p1.x && p.x < this.p2.x && p.y > this.p1.y && p.y < this.p2.y);
    }

    oncorner(p: Point): boolean {
        for (let pc of this.corners()) {
            if (p.manhat(pc) < CORNER_POINT_RADIUS)
                return true;
        }
        return false;
    }

    corners(): Array < Point > {
        let points: Array < Point > = new Array < Point > ();
        let x: number;
        let y: number;
        let list = [0.0, 0.5, 1.0];

        // Create 3x3 points
        for (let t1 of list) {
            x = Math.round((1. - t1) * this.p1.x + t1 * this.p2.x);
            for (let t2 of list) {
                y = Math.round((1.0 - t2) * this.p1.y + t2 * this.p2.y);
                points.push(new Point(x, y));
            }
        }
        return points;
    }

    height(): number {
        return Math.abs(this.p2.y - this.p1.y);
    }

    width(): number {
        return Math.abs(this.p2.x - this.p1.x);
    }
}

let p1 = new Point(10, 200);
let p2 = new Point(100, 20);
let rect = new Rectangle(p1, p2);
rect.draw("");
console.log("Rectangle corners: ", rect.corners());