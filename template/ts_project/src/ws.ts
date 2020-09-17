// Author: Dell Du, 2020-09-15

"use strict";

class Size {
    w: number;
    h: number;

    constructor() {
        this.w = this.h = 0;
    }
}

// enum Shape {
// 	Point,
// 	Rect,
// 	Circle,
// 	Ellipse,
// 	Polygon,
// 	Polyline
// }

class Point {
    x: number;
    y: number;

    constructor() {
        this.x = this.y = 0;
    }
}

class Rect {
    x: number;
    y: number;
    w: number;
    h: number;

    constructor() {
        this.x = this.y = this.w = this.h = 0;
    }
}

const POINT_RADIUS = 3;
const CONTROL_POINT_COLOR = "0xff0000";

class Workspcae {
    canvas: HTMLCanvasElement;
    context: any;
    zoom_scale: number;
    background: HTMLImageElement;

    points:Array<Point>;
    rects:Array<Rect>;

    constructor(canvas_id: string) {
        this.canvas = document.getElementById(canvas_id) as HTMLCanvasElement;
        this.context = this.canvas.getContext('2d');
        this.zoom_scale = 1.0;

       	this.points = new Array<Point>();
       	this.rects = new Array<Rect>();
    }

    setSize(size: Size) {
        this.canvas.width = size.w;
        this.canvas.height = size.h;
    }

    getSize(): Size {
        const w: number = this.canvas.width || 640;
        const h: number = this.canvas.height || 480;
        return { w, h }
    }

    drawControlPoint(cx:number, cy:number) {
        this.context.beginPath();
        this.context.arc(cx, cy, POINT_RADIUS, 0, 2 * Math.PI, false);
        this.context.closePath();
        this.context.fillStyle = CONTROL_POINT_COLOR;
        this.context.globalAlpha = 1.0;
        this.context.fill();
    }

    addPoint(p:Point) {
    	this.points.push(p);
    }

    findPoint(p:Point):number {
    	return this.points.lastIndexOf(p);
    }

    drawPoint(p:Point) {
        this.context.beginPath();
        this.context.arc(p.x, p.y, POINT_RADIUS, 0, 2 * Math.PI, false);
        this.context.closePath();
    }

    delPoint(i:number) {
    	this.points.splice(i, 1);	// here 1 means count
    }

    addRect(r:Rect) {
    	this.rects.push(r);
    }

    findRect(r:Rect):number {
    	return this.rects.lastIndexOf(r);
    }

    drawRect(r:Rect) {
        this.context.beginPath();
        this.context.moveTo(r.x, r.y);
        this.context.lineTo(r.x + r.w, r.y);
        this.context.lineTo(r.x + r.w, r.y + r.h);
        this.context.lineTo(r.x, r.y + r.h);
        this.context.closePath();
    }

    delRect(i:number) {
    	this.rects.splice(i, 1);	// here 1 means count
    }

    draw() {
        this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.context.scale(this.zoom_scale, this.zoom_scale);

        if (this.background) {
            this.context.drawImage(this.background, 0, 0, this.canvas.width, this.canvas.height);
        }
        for (let p of this.points)
        	this.drawPoint(p);
        for (let r of this.rects)
    		this.drawRect(r);
    }
}

function loadImage(ws: Workspcae, image: HTMLImageElement) {
    image.onload = function() {
        ws.background = image
    }
}

// Call demo:
// let ws = new Workspcae("image_clean_canvas");
// let size = ws.getSize();
// console.log(size.w, size.h);
// let image:HTMLImageElement = new Image()
// loadImage(ws, image)
// image.src = "waterfall.jpg";