// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

"use strict";

const CORNER_POINT_RADIUS = 3;

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
        // p -- this 
    }
}

abstract class Shape2d {
    name: string;
    constructor(name: string) {
        this.name = name;
    }

    abstract inside(p: Point): boolean; // p inside of rect
    abstract oncorner(p: Point): boolean; // p is on corner of rect

    abstract draw(brush: any);
}

class Rect extends Shape2d {
    p1: Point;
    p2: Point;

    constructor(p1: Point, p2: Point) {
        super("rect");

        // Normalization
        let x1, y1, x2, y2;
        if (p1.x < p2.x) {
            x1 = p1.x;
            x2 = p2.x;
        } else {
            x1 = p2.x;
            x2 = p1.x;
        }
        if (p1.y < p2.y) {
            y1 = p1.y;
            y2 = p2.y;
        } else {
            y1 = p2.y;
            y2 = p1.y;
        }

        this.p1 = new Point(x1, y1);
        this.p2 = new Point(x2, y2);
    }

    draw(brush: any) {
        console.log("Draw rect.");
    }

    inside(p: Point): boolean {
        return (p.x > this.p1.x && p.x < this.p2.x && p.y > this.p1.y && p.y < this.p2.y);
    }

    oncorner(p: Point): boolean {
        let x: number;
        let y: number;
        let list = [0.0, 0.5, 1.0];

        // Create 3x3 points
        for (let t1 of list) {
            x = Math.round((1. - t1) * this.p1.x + t1 * this.p2.x);
            for (let t2 of list) {
                y = Math.round((1.0 - t2) * this.p1.y + t2 * this.p2.y);
                if (p.manhat(new Point(x, y)) < CORNER_POINT_RADIUS)
                    return true;
            }
        }

        return false;
    }

    height(): number {
        return Math.abs(this.p1.y - this.p2.y);
    }

    width(): number {
        return Math.abs(this.p1.x - this.p2.x);
    }

}