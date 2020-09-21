// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

"use strict";

const DISTANCE_THRESHOLD = 2;
const VERTEX_COLOR = "0xff0000";

const enum ShapeID {
    Rectangle,
    Ellipse,
    Polygon,
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

    // euclid distance
    distance(p: Point): number {
        return Math.sqrt((p.x - this.x) * (p.x - this.x) + (p.y - this.y) * (p.y - this.y));
    }

    // point to line(p1 -- p2) distance
    tolineDistance(p1: Point, p2: Point): number {
        // p1 === p2
        if (p1.x === p2.x && p1.y === p2.y)
            return this.distance(p1);

        // d = |Ax + By + C|/sqrt(A*B + B*B), A=y2-y1, B=x1-x2, C=x2*y1 - x1*y2
        let A = p2.y - p1.y;
        let B = p1.x - p2.x;
        let C = p2.x * p1.y - p1.x * p2.y;
        let d = Math.abs(A * this.x + B * this.y + C);
        let s = Math.sqrt(A * A + B * B) + 0.000001;
        return d / s;
    }
    // < 0 if p lies on the left side of (p1, p2)
    // = 0 if p lies on the line of (p1, p2)
    // > 0 if p lies on the right side of (p1, p2)
    onLeft(p1: Point, p2: Point): number {
        // k(p1, p2) = (p2.y - p1.y)/(p2.x - p1.x)
        return (((p2.x - p1.x) * (this.y - p1.y)) - ((this.x - p1.x) * (p2.y - p1.y)));
    }
}

class Mouse {
    start: Point;
    moving: Point;
    stop: Point;

    constructor() {
        this.reset();
    }

    reset() {
        this.start = new Point(0, 0);
        this.moving = new Point(0, 0);
        this.stop = new Point(0, 0);
    }

    isclick(): boolean {
        let d = this.start.distance(this.stop);
        return d <= DISTANCE_THRESHOLD;
    }
}

abstract class Shape2d {
    id: ShapeID;

    constructor(id: ShapeID) {
        this.id = id;
    }

    abstract vertex(): Array < Point > ;
    abstract inside(p: Point): boolean; // p inside 2d shape object
    abstract onEdge(p: Point): boolean; // is p on edge of 2d shape object ?

    abstract draw(brush: any);

    // Reference
    onVertex(p: Point): boolean {
        for (let pc of this.vertex()) {
            if (p.distance(pc) <= DISTANCE_THRESHOLD)
                return true;
        }
        return false;
    }

    drawVertex(brush: any, p: Point) {
        brush.beginPath();
        brush.arc(p.x, p.y, DISTANCE_THRESHOLD, 0, 2 * Math.PI, false);
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

    onEdge(p: Point): boolean {
        let i = 0;
        let x = 0;
        let points = this.vertex();
        let len = points.length; // 4
        let d = p.tolineDistance(points[len - 1], points[0]);
        for (i = 0; i < len - 1; i++) {
            x = p.tolineDistance(points[i], points[i + 1]);
            if (x < d)
                d = x;
        }
        return d <= DISTANCE_THRESHOLD;
    }

    vertex(): Array < Point > {
        let points = new Array < Point > ();
        // x1y1, x2y1, x2y2, x1y2
        points.push(this.p1);
        points.push(new Point(this.p2.x, this.p1.y));
        points.push(this.p2);
        points.push(new Point(this.p1.x, this.p2.y));
        return points;
    }

    draw(brush: any) {
        // console.log("Shape ID:", this.id, ", ", this.p1, this.p2);
        brush.beginPath();
        brush.moveTo(this.p1.x, this.p1.y);
        brush.lineTo(this.p2.x, this.p1.y);
        brush.lineTo(this.p2.x, this.p2.y);
        brush.lineTo(this.p1.x, this.p2.y);
        brush.closePath();
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
        return (this.c.x - p.x) * (this.c.x - p.x) / (this.r.x * this.r.x) +
            (this.c.y - p.y) * (this.c.y - p.y) / (this.r.y * this.r.y) < 1;
    }

    onEdge(p: Point): boolean {
        let threshold = DISTANCE_THRESHOLD / this.r.x * this.r.y;
        let delta = (this.c.x - p.x) * (this.c.x - p.x) / (this.r.x * this.r.x) +
            (this.c.y - p.y) * (this.c.y - p.y) / (this.r.y * this.r.y) - 1.0;
        return Math.abs(delta) < threshold;
    }

    vertex(): Array < Point > {
        let points = new Array < Point > ();
        points.push(new Point(this.c.x - this.r.x, this.c.y));
        points.push(new Point(this.c.x, this.c.y + this.r.y));
        points.push(new Point(this.c.x + this.r.x, this.c.y));
        points.push(new Point(this.c.x, this.c.y - this.r.y));
        return points;
    }

    draw(brush: any) {
        brush.beginPath();
        brush.ellipse(this.c.x, this.c.y, this.r.x, this.r.y, 0, 0, 2 * Math.PI);
        brush.closePath();
    }
}

class Polygon extends Shape2d {
    points: Array < Point > ;

    constructor() {
        super(ShapeID.Polygon);
        this.points = new Array < Point > ();
    }

    inside(p: Point): boolean {
        let n = this.points.length;
        if (n < 3)
            return false;

        let vset = this.points.slice(0); // Deep copy
        vset.push(this.points[0]); // Closed
        let wn = 0; // the  winding number counter
        for (let i = 0; i < n; i++) {
            var is_left = p.onLeft(vset[i], vset[i + 1]);
            if (p.y >= vset[i].y) {
                // P1.y <= p.y < P2.y
                if (p.y < vset[i + 1].y && is_left > 0) ++wn;
            } else {
                // P2.y <= p.y < P1.y
                if (p.y >= vset[i + 1].y && is_left < 0) --wn;
            }
        }
        return (wn === 0) ? false : true;
    }

    onEdge(p: Point): boolean {
        let i = 0;
        let x = 0;
        let len = this.points.length;
        if (len < 2)
            return false;
        let d = p.tolineDistance(this.points[len - 1], this.points[0]);
        for (i = 0; i < len - 1; i++) {
            x = p.tolineDistance(this.points[i], this.points[i + 1]);
            if (x < d)
                d = x;
        }
        return d <= DISTANCE_THRESHOLD;
    }

    vertex(): Array < Point > {
        return this.points;
    }

    draw(brush: any) {
        // console.log("Shape ID:", this.id, ", ", this.points);
        if (this.points.length < 3)
            return;
        brush.beginPath();
        brush.moveTo(this.points[0].x, this.points[0].y);
        for (let i = 0; i < this.points.length; ++i)
            brush.lineTo(this.points[i].x, this.points[i].y);
        brush.lineTo(this.points[0].x, this.points[0].y); // close loop
        brush.stroke();
    }

    push(p: Point) {
        this.points.push(p);
    }

    insert(index: number, p: Point) {
        this.points.splice(index, 0, p);
    }

    delete(index: number) {
        this.points.slice(index, 1);
    }

    pop() {
        this.points.pop();
    }
}

let p1 = new Point(10, 200);
let p2 = new Point(100, 20);
let rect = new Rectangle(p1, p2);
console.log("Rectangle vertex: ", rect.vertex());

let e = new Ellipse(p1, p2);
console.log("Ellipse vertex:", e.vertex());

let poly = new Polygon();
poly.push(new Point(20, 10));
poly.push(new Point(10, 25));
poly.push(new Point(30, 30));
poly.push(new Point(50, 20));
poly.push(new Point(40, 0));

let a = new Array < Shape2d > ();
a.push(e);
a.push(rect);
a.push(poly);

console.log("---------------------------------------------------");
console.log(a);

console.log("for all sets ---------------------------------------------------");
for (let x of a) {
    console.log(x.id, "---->", x);
}

let x = new Point(25, 25);
console.log(x, "is inside ?", poly, poly.inside(x));

x = new Point(100, 25);
console.log(x, "is inside ?", poly, poly.inside(x));
poly.insert(13, new Point(110, 35));
console.log(x, "is inside ?", poly, poly.inside(x));
poly.delete(13);
console.log("delete index 13 ?", poly, poly.inside(x));