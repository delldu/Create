// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

"use strict";

const VERTEX_RADIUS = 3;
const VERTEX_COLOR = "0xff0000";

const enum ShapeID {
    Rectangle,
    Ellipse,
    Polygon,
    Polyline,
    Cube,
    Cone,
    Cylinder,
    Sphere
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

    abstract vertex(): Array < Point > ;
    abstract inside(p: Point): boolean; // p inside 2d shape object
    abstract draw(brush: any);

    // Reference
    onVertex(p: Point): boolean {
        for (let pc of this.vertex()) {
            if (p.manhat(pc) < VERTEX_RADIUS)
                return true;
        }
        return false;
    }

    drawVertex(brush: any, p: Point) {
        brush.beginPath();
        brush.arc(p.x, p.y, VERTEX_RADIUS, 0, 2 * Math.PI, false);
        brush.closePath();
        brush.fillStyle = VERTEX_COLOR;
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

    inside(p: Point): boolean {
        return (p.x > this.p1.x && p.x < this.p2.x && p.y > this.p1.y && p.y < this.p2.y);
    }

    vertex(): Array < Point > {
        let points: Array < Point > = new Array < Point > ();
        let x: number;
        let y: number;
        let list = [0.0, 1.0];

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

    draw(brush: any) {
        console.log("Shape ID:", this.id, ", ", this.p1, this.p2);
        // brush.beginPath();
        // brush.moveTo(this.p1.x, this.p1.y);
        // brush.lineTo(this.p2.x, this.p1.y);
        // brush.lineTo(this.p2.x, this.p2.y);
        // brush.lineTo(this.p1.x, this.p2.y);
        // brush.closePath();
    }

    height(): number {
        return Math.abs(this.p2.y - this.p1.y);
    }

    width(): number {
        return Math.abs(this.p2.x - this.p1.x);
    }
}

class Ellipse extends Shape2d {
    c: Point; // Center
    r: Point; // Radius of x, y

    constructor(c: Point, r: Point) {
        super(ShapeID.Ellipse);
        this.c = c;
        this.r = r;
    }

    inside(p: Point): boolean {
        return (this.c.x - p.x) * (this.c.x - p.x) / (this.r.x * this.r.x) + (this.c.y - p.y) * (this.c.y - p.y) / (this.r.y * this.r.y) < 1;
    }

    vertex(): Array < Point > {
        let points: Array < Point > = new Array < Point > ();
        points.push(new Point(this.c.x - this.r.x, this.c.y));
        points.push(new Point(this.c.x + this.r.x, this.c.y));
        points.push(new Point(this.c.x, this.c.y - this.r.y));
        points.push(new Point(this.c.x, this.c.y + this.r.y));
        return points;
    }

    draw(brush: any) {
        console.log("Shape ID:", this.id, ", ", this.c, this.r);
        // brush.beginPath();
        // brush.moveTo(this.p1.x, this.p1.y);
        // brush.lineTo(this.p2.x, this.p1.y);
        // brush.lineTo(this.p2.x, this.p2.y);
        // brush.lineTo(this.p1.x, this.p2.y);
        // brush.closePath();
    }
}

class Polygon extends Shape2d {
    points: Array < Point > ;

    constructor() {
        super(ShapeID.Polygon);
        this.points = new Array < Point > ();
        // this.points.push(new Point(0, 0));
        // this.points.push(new Point(10, 10));
        // this.points.push(new Point(30, 30));
        // this.points.push(new Point(80, 80));
    }

    inside(p: Point): boolean {
        return false;
    }

    vertex(): Array < Point > {
        return this.points;
    }

    draw(brush: any) {
        console.log("Shape ID:", this.id, ", ", this.points);
        // brush.beginPath();
        // brush.moveTo(this.p1.x, this.p1.y);
        // brush.lineTo(this.p2.x, this.p1.y);
        // brush.lineTo(this.p2.x, this.p2.y);
        // brush.lineTo(this.p1.x, this.p2.y);
        // brush.closePath();
    }
}

class Polyline extends Shape2d {
    points: Array < Point > ;

    constructor() {
        super(ShapeID.Polyline);
        this.points = new Array < Point > ();
        // this.points.push(new Point(0, 0));
        // this.points.push(new Point(10, 10));
        // this.points.push(new Point(30, 30));
        // this.points.push(new Point(80, 80));
    }

    inside(p: Point): boolean {
        return false;
    }

    vertex(): Array < Point > {
        return this.points;
    }

    draw(brush: any) {
        console.log("Shape ID:", this.id, ", ", this.points);
        // brush.beginPath();
        // brush.moveTo(this.p1.x, this.p1.y);
        // brush.lineTo(this.p2.x, this.p1.y);
        // brush.lineTo(this.p2.x, this.p2.y);
        // brush.lineTo(this.p1.x, this.p2.y);
        // brush.closePath();
    }
}


let p1 = new Point(10, 200);
let p2 = new Point(100, 20);
let rect = new Rectangle(p1, p2);
rect.draw("");
console.log("Rectangle vertex: ", rect.vertex());

let e = new Ellipse(p1, p2);
console.log("Ellipse vertex:", e.vertex());
e.draw("");


let p = new Polyline();
console.log("Polyline vertex:", p.vertex());
p.draw("");