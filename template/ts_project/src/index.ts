// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

"use strict";

class Message {
    timer: number;
    element: HTMLElement; // message element

    constructor(id: string) {
        this.timer = 0;
        this.element = document.getElementById(id);
        this.element.addEventListener('dblclick', function() {
            this.style.display = 'none';
        }, false);
    }

    show(msg: string, t: number) {
        if (this.timer) {
            clearTimeout(this.timer);
        }
        this.element.innerHTML = msg;
        this.timer = setTimeout(() => {
            this.element.style.display = 'none';
        }, t);
    }
}

// Test:
// <div id="message_id" class="message">Message Bar</div>
// function load() {
//     msgbar = new Message("message_id");
//     msgbar.show("This is a message ........................", 10000); // 10 s
// }// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

"use strict";

class Progress {
    element: HTMLElement;

    constructor(i:number) {
        this.element = document.getElementsByTagName("progress")[0];
    }

    update(v: number) {
        this.element.setAttribute('value', v.toString());
    }

    startDemo(value:number) {
        value = value + 1;
        this.update(value);
        if (value < 100) {
            setTimeout(()=>{this.startDemo(value);}, 20);
        }
    }
}

// Test:
// <progress value="0" max="100">Progress Bar</div>
// function load() {
//     pb = new Progress(0);
//     pb.startDemo(0); // start 0
// }
// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

"use strict";

const DISTANCE_THRESHOLD = 2;
const EDGE_LINE_WIDTH = 1.0;
const VERTEX_COLOR = "#ff0000";

const ZOOM_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4, 5, 6, 7, 8, 9, 10];
const DEFAULT_ZOOM_LEVEL = 3; // 1.0 index

// 2D shape id
const enum ShapeID {
    None,
    Rectangle,
    Ellipse,
    Polygon
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
    abstract draw(brush: CanvasRenderingContext2D, selected: boolean);

    // Reference
    onVertex(p: Point): boolean {
        for (let pc of this.vertex()) {
            if (p.distance(pc) <= DISTANCE_THRESHOLD)
                return true;
        }
        return false;
    }

    drawVertex(brush: CanvasRenderingContext2D) {
        brush.save();
        brush.fillStyle = VERTEX_COLOR;
        brush.globalAlpha = 1.0;
        for (let p of this.vertex()) {
            brush.beginPath();
            brush.arc(p.x, p.y, DISTANCE_THRESHOLD, 0, 2 * Math.PI, false);
            brush.closePath();
            brush.fill();
        }
        brush.restore();
    }
}

class Rectangle extends Shape2d {
    p1: Point;
    p2: Point;

    constructor(p1: Point, p2: Point) {
        super(ShapeID.Rectangle);
        this.p1 = new Point(p1.x, p1.y);
        this.p2 = new Point(p2.x, p2.y); // Deep copy

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

    draw(brush: CanvasRenderingContext2D, selected: boolean) {
        brush.save();
        if (selected) {
            brush.fillStyle = VERTEX_COLOR;
            brush.globalAlpha = 1.0;
        }
        let points = this.vertex();
        brush.beginPath();
        brush.moveTo(points[0].x, points[0].y);
        for (let i = 0; i < points.length; ++i)
            brush.lineTo(points[i].x, points[i].y);
        brush.lineTo(points[0].x, points[0].y); // close loop
        brush.closePath();
        if (selected) {
            brush.fill();
            this.drawVertex(brush);
        } else {
            brush.stroke();
        }
        brush.restore();
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
        this.c = new Point(c.x, c.y);
        this.r = new Point(r.x, r.y); // Dot share with others !
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

    draw(brush: CanvasRenderingContext2D, selected: boolean) {
        brush.save();
        if (selected) {
            brush.fillStyle = VERTEX_COLOR;
            brush.globalAlpha = 1.0;
        }
        brush.beginPath();
        brush.ellipse(this.c.x, this.c.y, this.r.x, this.r.y, 0, 0, 2 * Math.PI);
        brush.closePath();
        if (selected) {
            brush.fill();
            this.drawVertex(brush);
        } else {
            brush.stroke();
        }
        brush.restore();
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

    draw(brush: CanvasRenderingContext2D, selected: boolean) {
        // console.log("Shape ID:", this.id, ", ", this.points);
        if (this.points.length < 1)
            return;
        brush.save();
        if (selected) {
            brush.fillStyle = VERTEX_COLOR;
            brush.globalAlpha = 1.0;
        }
        brush.beginPath();
        brush.moveTo(this.points[0].x, this.points[0].y);
        for (let i = 0; i < this.points.length; ++i)
            brush.lineTo(this.points[i].x, this.points[i].y);
        brush.lineTo(this.points[0].x, this.points[0].y); // close loop
        brush.closePath();
        if (selected) {
            brush.fill();
            this.drawVertex(brush);
        } else {
            brush.stroke();
        }
        brush.restore();
    }

    push(p: Point) {
        this.points.push(new Point(p.x, p.y)); // Dot share with others, or will be a disater !!!
    }

    insert(index: number, p: Point) {
        this.points.splice(index, 0, new Point(p.x, p.y));
    }

    delete(index: number) {
        this.points.slice(index, 1);
    }

    pop() {
        this.points.pop();
    }
}

class Canvas {
    canvas: HTMLCanvasElement; // canvas element
    private brush: CanvasRenderingContext2D;

    // Shape container
    drawing_shape: ShapeID; // if == ShapeID.None is view mode
    private regions: Array < Shape2d > ; // shape regions
    private selected_index: number;
    private drawing_polygon: Polygon;   // this is temperay record

    // Zoom control
    zoom_index: number;

    // Handle mouse, keyboard device
    private mouse: Mouse;

    constructor(id: string) {
        this.canvas = document.getElementById(id) as HTMLCanvasElement;
        this.brush = this.canvas.getContext('2d') as CanvasRenderingContext2D;

        this.drawing_shape = ShapeID.None;
        this.regions = new Array < Shape2d > ();
        this.drawing_polygon = new Polygon();
        this.selected_index = -1;

        // Line width and color
        this.brush.strokeStyle = VERTEX_COLOR;
        this.brush.lineWidth = EDGE_LINE_WIDTH;

        this.zoom_index = DEFAULT_ZOOM_LEVEL;

        this.mouseInitialize();
    }

    setMessage(message: string) {
        console.log(message);
    }

    setShape(shape: ShapeID) {
        if (shape == ShapeID.None || shape == ShapeID.Rectangle || shape == ShapeID.Ellipse || shape == ShapeID.Polygon) {
            this.drawing_shape = shape;
            // Clear Polygon for shape change
            if (shape != ShapeID.Polygon) {
                this.drawing_polygon = new Polygon();
            }
            return;
        }
        this.setMessage("Canvas only support None/Rectangle/Ellipse/Polygon shape.");
    }

    private viewModeMouseClickHandler(e: MouseEvent) {
        if (this.drawing_shape != ShapeID.None)
            return;
        if (! this.mouse.isclick())
            return;

    }

    private viewModeMouseMovingHandler(e: MouseEvent) {
        if (this.drawing_shape != ShapeID.None)
            return;
        if (this.mouse.isclick())
            return;
        
    }

    private editModeMouseClickHandler(e: MouseEvent) {
        if (this.drawing_shape == ShapeID.None)
            return;
        if (! this.mouse.isclick())
            return;

    }

    private editModeMouseMovingHandler(e: MouseEvent) {
        if (this.drawing_shape == ShapeID.None)
            return;
        if (this.mouse.isclick())
            return;
    }

    private mouseInitialize() {
        this.mouse = new Mouse();

        this.canvas.addEventListener('mousedown', (e: MouseEvent) => {
            this.mouse.start.x = e.offsetX;
            this.mouse.start.y = e.offsetY;
        }, false);
        this.canvas.addEventListener('mouseup', (e: MouseEvent) => {
            this.mouse.stop.x = e.offsetX;
            this.mouse.stop.y = e.offsetY;
        }, false);
        this.canvas.addEventListener('mouseover', (e: MouseEvent) => {
            this.redraw();
        }, false);
        this.canvas.addEventListener('mousemove', (e: MouseEvent) => {
            this.mouse.moving.x = e.offsetX;
            this.mouse.moving.y = e.offsetY;
        }, false);
        this.canvas.addEventListener('wheel', (e: WheelEvent) => {
            if (e.ctrlKey) {
                if (e.deltaY < 0) {
                    this.setZoom(this.zoom_index + 1);
                } else {
                    this.setZoom(this.zoom_index - 1);
                }
                e.preventDefault();
            }
        }, false);

        // Canvas general mouse handler
        this.canvas.addEventListener('mouseup', (e: MouseEvent) => { this.viewModeMouseClickHandler(e); }, false);
        this.canvas.addEventListener('mouseup', (e: MouseEvent) => { this.editModeMouseClickHandler(e); }, false);
        this.canvas.addEventListener('mouseup', (e: MouseEvent) => { this.viewModeMouseMovingHandler(e); }, false);
        this.canvas.addEventListener('mouseup', (e: MouseEvent) => { this.editModeMouseMovingHandler(e); }, false);
        
        // touch screen event handlers, TODO: test !!!
        // this.canvas.addEventListener('touchstart', (e) => {
        //     this.mouse.start.x = e.offsetX;
        //     this.mouse.start.y = e.offsetY;
        // }, false);
        // this.canvas.addEventListener('touchend', (e) => {
        //     this.mouse.stop.x = e.offsetX;
        //     this.mouse.stop.y = e.offsetY;
        // }, false);
        // this.canvas.addEventListener('touchmove', (e) => {
        //     this.mouse.moving.x = e.offsetX;
        //     this.mouse.moving.y = e.offsetY;
        // }, false);
    }

    setZoom(index: number) {
        index = Math.round(index);
        if (index < 0)
            index = 0;
        if (index >= ZOOM_LEVELS.length)
            index = ZOOM_LEVELS.length - 1;
        this.zoom_index = index;
        this.brush.scale(ZOOM_LEVELS[this.zoom_index], ZOOM_LEVELS[this.zoom_index]);
    }

    redraw() {
        this.brush.clearRect(0, 0, this.canvas.width, this.canvas.height);
        // Draw image ...

        // Draw regions ...
        for (let i in this.regions) {
            if (parseInt(i) === this.selected_index)
                this.regions[i].draw(this.brush, true);
            else
                this.regions[i].draw(this.brush, false);
        }

        // Draw drawing polygon ...
        if (this.drawing_shape === ShapeID.Polygon) {
            this.drawing_polygon.draw(this.brush, false);
        }
    }

    pushShape(s: Shape2d) {
        this.regions.push(s);
    }

    Delete(index: number) {
        this.regions.slice(index, 1);
    }

    popShape() {
        this.regions.pop();
    }

    private test() {
        let p1 = new Point(10, 10);
        let p2 = new Point(320, 240);
        this.pushShape(new Rectangle(p1, p2));
        console.log("-------------------------->", this.regions);

        p1.x = 320;
        p1.y = 240;
        p2.x = 300;
        p2.y = 220;
        this.pushShape(new Ellipse(p1, p2));

        let polygon = new Polygon();
        polygon.push(new Point(10, 240));
        polygon.push(new Point(310, 470));
        polygon.push(new Point(630, 240));
        polygon.push(new Point(540, 160));
        polygon.push(new Point(120, 30));
        this.pushShape(polygon);
    }
}// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************


"use strict";

// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************


"use strict";

