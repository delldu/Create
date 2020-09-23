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

const MODE_LIST = [ShapeID.None, ShapeID.Rectangle, ShapeID.Ellipse, ShapeID.Polygon];

class Point {
    x: number;
    y: number;
    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }

    move(deltaX: number, deltaY: number) {
        this.x += deltaX;
        this.y += deltaY;
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

    pressed: boolean;
    draged: boolean;
    start_drawing: boolean;

    constructor() {
        this.reset();
    }

    reset() {
        this.start = new Point(0, 0);
        this.moving = new Point(0, 0);
        this.stop = new Point(0, 0);

        this.pressed = false;
        this.draged = false;
        this.start_drawing = false;
    }

    isclick(): boolean {
        let d = this.start.distance(this.stop);
        // console.log("mouse isclick? ", "start = ", this.start, "stop = ", this.stop, "distance: ", d);
        return d <= DISTANCE_THRESHOLD;
    }
}

abstract class Shape2d {
    id: ShapeID;

    constructor(id: ShapeID) {
        this.id = id;
    }

    abstract vertex(): Array < Point > ;

    // Search ...
    abstract bodyHas(p: Point): boolean; // p inside 2d shape ?
    abstract whichEdgeHas(p: Point): number; // which edge include point p ?
    // which vertex include point p ?
    whichVertexHas(p: Point): number {
        let vset = this.vertex();
        let n = vset.length;
        for (let i = 0; i < n; i++) {
            if (p.distance(vset[i]) <= DISTANCE_THRESHOLD)
                return i;
        }
        return -1;
    }

    abstract draw(brush: CanvasRenderingContext2D, selected: boolean);

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

    move(deltaX: number, deltaY: number) {
        for (let p of this.vertex()) {
            p.x += deltaX;
            p.y += deltaY;
        }
    }

    // Vertex add/delete
    abstract push(p: Point);
    abstract insert(index: number, p: Point);
    abstract delete(index: number);
    abstract pop();
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

    bodyHas(p: Point): boolean {
        return (p.x > this.p1.x && p.x < this.p2.x && p.y > this.p1.y && p.y < this.p2.y);
    }

    // useless, so just return -1 to meet interface
    whichEdgeHas(p: Point): number {
        return -1;
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

    // Vertex add/delete
    push(p: Point) {}
    insert(index: number, p: Point) {}
    delete(index: number) {}
    pop() {}
}

class Ellipse extends Shape2d {
    c: Point; // Center
    r: Point; // Radius of x, y

    constructor(c: Point, r: Point) {
        super(ShapeID.Ellipse);
        this.c = new Point(c.x, c.y);
        this.r = new Point(r.x, r.y); // Dot share with others !
    }

    bodyHas(p: Point): boolean {
        return (this.c.x - p.x) * (this.c.x - p.x) / (this.r.x * this.r.x) +
            (this.c.y - p.y) * (this.c.y - p.y) / (this.r.y * this.r.y) < 1;
    }

    // Useless, just return -1 to meet interface
    whichEdgeHas(p: Point): number {
        return -1;
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

    // Vertex add/delete
    push(p: Point) {}
    insert(index: number, p: Point) {}
    delete(index: number) {}
    pop() {}
}

class Polygon extends Shape2d {
    points: Array < Point > ;

    constructor() {
        super(ShapeID.Polygon);
        this.points = new Array < Point > ();
    }

    bodyHas(p: Point): boolean {
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

    // Support add point on edge in edit mode ...
    whichEdgeHas(p: Point): number {
        let n = this.points.length;
        if (n < 3)
            return -1;
        let vset = this.points.slice(0); // Deep copy
        vset.push(this.points[0]); // Closed
        for (let i = 0; i < n; i++) {
            if (p.tolineDistance(vset[i], vset[i + 1]) <= DISTANCE_THRESHOLD)
                return i;
        }
        return -1;
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
        this.points.push(new Point(p.x, p.y)); // Dot share, or will be disater !!!
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

// Mini Canvas: only redraw single object
class MinCanvas {
    canvas: HTMLCanvasElement;
    brush: CanvasRenderingContext2D;

    constructor(id: string) {
        this.canvas = document.getElementById(id) as HTMLCanvasElement;
        this.brush = this.canvas.getContext('2d') as CanvasRenderingContext2D;
    }

    draw(shape: Shape2d) {
        this.brush.clearRect(0, 0, this.canvas.width, this.canvas.height);
        shape.draw(this.brush, false);
    }
}

class Canvas {
    canvas: HTMLCanvasElement; // canvas element
    private brush: CanvasRenderingContext2D;

    private backgroud: HTMLImageElement;
    private drawing_canvas: HTMLCanvasElement; // This is a temperary canvas

    mode_index: number;

    // Shape container
    private regions: Array < Shape2d > ; // shape regions
    private selected_index: number;
    private drawing_polygon: Polygon; // this is temperay record

    // Zoom control
    zoom_index: number;

    // Handle mouse, keyboard device
    private mouse: Mouse;

    constructor(id: string) {
        /*
        #image_panel        { position:relative; outline:none; }
        #image_panel img    { visibility:hidden; opacity:0; position:absolute; top:0px; left:0px; width:100%; height:100%; outline:none; }
        #image_panel canvas { position:absolute; top:0px; left:0px; outline:none;}
        #image_panel .visible { visibility:visible !important; opacity:1 !important; }
        */
        this.canvas = document.getElementById(id) as HTMLCanvasElement;
        this.canvas.style.position = "absolute";
        this.canvas.style.opacity = "0.5";
        // this.canvas.style.zIndex = "-1";
        this.canvas.style.top = "5px";
        this.canvas.style.left = "5px";
        this.canvas.style.border = "1px solid green";

        this.brush = this.canvas.getContext('2d') as CanvasRenderingContext2D;
        this.regions = new Array < Shape2d > ();
        this.drawing_polygon = new Polygon();
        this.selected_index = -1;

        // Line width and color
        this.brush.strokeStyle = VERTEX_COLOR;
        this.brush.lineWidth = EDGE_LINE_WIDTH;

        this.zoom_index = DEFAULT_ZOOM_LEVEL;

        this.mode_index = 0;

        this.registerEventHandlers();

        // Create backgroud and drawing canvas
        let parent = this.canvas.parentElement;
        this.backgroud = document.createElement('img');
        this.backgroud.id = this.canvas.id + "_backgroud_image";
        this.backgroud.style.opacity = "1";
        // this.backgroud.style.zIndex = "0";
        this.backgroud.style.position = "absolute";
        this.backgroud.style.top = "0";
        this.backgroud.style.left = "0";
        this.backgroud.style.border = "1px solid red";
        this.backgroud.width = this.canvas.width;
        this.backgroud.height = this.canvas.height;
        parent.insertBefore(this.backgroud, this.canvas);
        this.backgroud.src = "dog.jpg";
        this.backgroud.onload = function() {

        }

        this.drawing_canvas = document.createElement('canvas');
        this.drawing_canvas.id = this.canvas.id + "_drawing_canvas";
        this.drawing_canvas.width = this.canvas.width;
        this.drawing_canvas.height = this.canvas.height;
        // this.drawing_canvas.style.zIndex = "0";
        this.drawing_canvas.style.position = "absolute";
        this.drawing_canvas.style.opacity = "0.5";
        this.drawing_canvas.style.top = "10px";
        this.drawing_canvas.style.left = "10px";
        this.drawing_canvas.style.border = "1px solid blue";
        parent.insertBefore(this.drawing_canvas, this.canvas);
    }

    setMessage(message: string) {
        console.log(message);
    }

    setMode(index: number) {
        // Bad case: (-1 % MODE_LIST.length) == -1
        if (index < 0)
            index = MODE_LIST.length;
        index = index % MODE_LIST.length;
        this.mode_index = index;
        console.log("Now canvas mode has been switched to ", this.mode_index);
    }

    getMode(): number {
        return this.mode_index;
    }

    isEditMode(): boolean {
        return this.mode_index > 0;
    }

    getShape(): ShapeID {
        return MODE_LIST[this.mode_index];
    }

    private viewModeMouseDownHandler(e: MouseEvent) {
        console.log("viewModeMouseDownHandler ...");
    }

    private viewModeMouseMoveHandler(e: MouseEvent) {
        // make sure moving distance is enough ...
        if (this.mouse.pressed && ! this.mouse.isclick()) {
            // todo ?
            console.log("viewModeMouseMove ... Draging ... moving backgroud for more deteails ?");
            this.canvas.style.cursor = "pointer";
            // this.canvas.start = new Point(this.mouse.stop.x, this.mouse.stop.y);
        }
    }

    private viewModeMouseUpHandler(e: MouseEvent) {
        console.log("viewModeMouseUpHandler ...");
        this.canvas.style.cursor = "default";
    }

    // private editModeMouseClickHandler(e: MouseEvent) {
    //     if (this.drawing_shape == ShapeID.None)
    //         return;

    //     if (! this.mouse.isclick())
    //         return;

    //     // Selected or remove object
    //     let [index, sub_index] = this.findInside(this.mouse.start);

    //     if (index >= 0) {
    //         if (index == this.selected_index)
    //             this.selected_index = -1;
    //         else
    //             this.selected_index = index;
    //         this.redraw();
    //         return;
    //     }

    //     // click on vertex/edge of selected pylogon, add remove or add point 
    //     if (this.selected_index >= 0 && this.drawing_shape == ShapeID.Polygon) {
    //         // Remove vertex
    //         [index, sub_index] = this.findVertex((this.mouse.start));
    //         if (index == this.selected_index) {
    //             this.regions[index].delete(sub_index);
    //             this.redraw();
    //             return;
    //         }
    //         // Add new vertex
    //         [index, sub_index] = this.findEdge((this.mouse.start));
    //         if (index == this.selected_index) {
    //             this.regions[index].insert(sub_index, this.mouse.start);
    //             this.redraw();
    //             return;
    //         }
    //         return;
    //     }

    //     // is drawing polygon on blank space ?
    //     if (this.selected_index < 0 && this.drawing_shape == ShapeID.Polygon) {
    //         this.drawing_polygon.push(this.mouse.start);
    //         this.redraw();
    //         return;
    //     }
    // }

    private editModeMouseDownHandler(e: MouseEvent) {
        // Start:  On vertext, On Edge, Inside, Blank area
        console.log("if click on blank --- this.mouse.start_drawing = true else this.mouse.start_drawing = false");
    }

    private editModeMouseMoveHandler(e: MouseEvent) {
        // make sure moving distance is enough ...
        if (this.mouse.pressed && ! this.mouse.isclick()) {
            console.log("editModeMouseMove ... Draging ... show MinCanvas with box ...");
        }


        // if (this.drawing_shape == ShapeID.None) {
        //     console.log("Moving_001 ...")
        //     return;
        // }

        // if (this.mouse.isclick()) {
        //     console.log("Moving_002 ...")
        //     return;
        // }

        // // Drag and drop on blank area ? Only for rectangle/ellipse drawing
        // if (this.selected_index < 0) {
        //     if (this.drawing_shape == ShapeID.Rectangle) {
        //         this.pushShape(new Rectangle(this.mouse.start, this.mouse.stop));
        //         this.redraw();
        //         return;
        //     }
        //     if (this.drawing_shape == ShapeID.Ellipse) {
        //         this.pushShape(new Ellipse(this.mouse.start, this.mouse.stop));
        //         this.redraw();
        //         return;
        //     }
        //     return;
        // }

        // // Now selected_index >= 0
        // // Resize object ?
        // let [index, sub_index] = this.findVertex(this.mouse.start);
        // if (index == this.selected_index) {
        //     let mx = this.mouse.stop.x - this.mouse.start.x;
        //     let my = this.mouse.stop.y - this.mouse.stop.y;
        //     let vpoint = this.regions[index].vertex();
        //     vpoint[sub_index].move(mx, my);
        //     this.redraw();
        //     return;
        // }
        // [index, sub_index] = this.findInside(this.mouse.start);
        // if (index == this.selected_index) {
        //     // Moving whole object
        //     let mx = this.mouse.stop.x - this.mouse.start.x;
        //     let my = this.mouse.stop.y - this.mouse.stop.y;
        //     this.regions[index].move(mx, my);
        //     return;
        // }
    }

    private viewModeMouseDblclickHandler(e: MouseEvent) {
        console.log("viewModeMouseDblclickHandler ...");
    }

    private editModeMouseDblclickHandler(e: MouseEvent) {
        console.log("editModeMouseDblclickHandler ...");

        //     if (this.drawing_shape != ShapeID.Polygon)
        //         return;

        //     // End drawing polygon
        //     if (this.drawing_polygon.vertex().length >= 3) {
        //         this.pushShape(this.drawing_polygon);
        //     }
        //     this.drawing_polygon = new Polygon();
        //     this.redraw();
    }

    private editModeMouseUpHandler(e: MouseEvent) {
        console.log("editModeMouseUpHandler ...");
        if (this.mouse.pressed && this.mouse.draged) {
            if (this.mouse.start_drawing) {
                console.log("Drawing is ok ..., for rectangle/ellipse, save result and close miniCanvas, for polygon ...");
                // we get new object, save them, redraw, ...
            } else {
                console.log("Set selected flags covered by box ..., close MinCanvas, reset mouse, redraw big canvas");
            }
            // this.mouse_start_drawing && this.getShape is polygon, we should reserver mouse history else we need this.mouse.reset();
        }
    }

    private viewModeKeyDownHandler(e: KeyboardEvent) {
        // console.log("viewModeKeydownHandler ...");
        e.preventDefault();
    }

    private viewModeKeyUpHandler(e: KeyboardEvent) {
        console.log("viewModeKeyUpHandler ...", e.key);

        if (e.key === "+") {
            this.setZoom(this.zoom_index + 1);
        } else if (e.key === "=") {
            this.setZoom(DEFAULT_ZOOM_LEVEL);
        } else if (e.key === "-") {
            this.setZoom(this.zoom_index - 1);
        } else if (e.key === 'F1') { // F1 for help
            // todo
        } else {
            // todo
        }
        e.preventDefault();
    }


    private editModeKeyDownHandler(e: KeyboardEvent) {
        console.log("editModeKeyDown ...", e.key);

        e.preventDefault();
    }

    private editModeKeyUpHandler(e: KeyboardEvent) {
        console.log("editModeKeyUpHandler ...", e.key);
        // if (e.key === "d") {
        //     delete selected objects ..., redraw ...
        //     return;
        // }
        // if (e.key === 'Enter' || e.key === 'Escape') {
                // end polygon drawing
        // }

        // if (e.key === 'Escape') {
        //     this.selected_index = -1;
        //     this.redraw();
        //     return;
        // }

        // if (e.key === 'Enter') {
        //     if (this.drawing_shape == ShapeID.Polygon) {
        //         // End drawing polygon
        //         if (this.drawing_polygon.vertex().length >= 3) {
        //             this.pushShape(this.drawing_polygon);
        //         }
        //         this.drawing_polygon = new Polygon();
        //         this.redraw();
        //         return;
        //     }
        // }

        // if (e.key === 'Backspace') {
        //     if (this.drawing_shape == ShapeID.Polygon) {
        //         // delete last vertex from polygon
        //         this.drawing_polygon.pop();
        //         this.redraw();
        //         return;
        //     }
        // }
        e.preventDefault();
    }

    private registerEventHandlers() {
        this.mouse = new Mouse();

        this.canvas.addEventListener('mousedown', (e: MouseEvent) => {
            // console.log("mousedown_001 ...", this.mouse.start);

            this.mouse.start.x = e.offsetX;
            this.mouse.start.y = e.offsetY;
            this.mouse.pressed = true;

            if (this.isEditMode())
                this.editModeMouseDownHandler(e);
            else
                this.viewModeMouseDownHandler(e);
        }, false);
        this.canvas.addEventListener('mouseup', (e: MouseEvent) => {
            // console.log("mousedown_002 ...", this.mouse.start);

            this.mouse.pressed = false;
            this.mouse.stop.x = e.offsetX;
            this.mouse.stop.y = e.offsetY;

            if (this.isEditMode())
                this.editModeMouseUpHandler(e);
            else
                this.viewModeMouseUpHandler(e);
            e.stopPropagation();
        }, false);
        this.canvas.addEventListener('mouseover', (e: MouseEvent) => {
            // console.log("mousedown_003 ...", this.mouse.start);
            this.redraw();
        }, false);
        this.canvas.addEventListener('mousemove', (e: MouseEvent) => {
            // console.log("mousedown_004 ...", this.mouse.start);
            this.mouse.moving.x = e.offsetX;
            this.mouse.moving.y = e.offsetY;
            this.mouse.draged = true;

            if (this.isEditMode())
                this.editModeMouseMoveHandler(e);
            else
                this.viewModeMouseMoveHandler(e);

            // // Drawing response ...
            // if (this.drawing_shape == ShapeID.Rectangle && this.mouse.pressed) {
            //     let rect = new Rectangle(this.mouse.start, this.mouse.moving);
            //     rect.draw(this.brush, false);
            //     return;
            // }
            // if (this.drawing_shape == ShapeID.Ellipse && this.mouse.pressed) {
            //     let ellipse = new Ellipse(this.mouse.start, this.mouse.moving);
            //     ellipse.draw(this.brush, false);
            //     return;
            // }

        }, false);
        this.canvas.addEventListener('wheel', (e: WheelEvent) => {
            if (e.ctrlKey) {
               // console.log("mousedown_005 ...", this.mouse.start);
                if (e.deltaY < 0) {
                    this.setZoom(this.zoom_index + 1);
                } else {
                    this.setZoom(this.zoom_index - 1);
                }
                e.preventDefault();
            }
        }, false);

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

        this.canvas.addEventListener('dblclick', (e: MouseEvent) => {
            if (this.isEditMode())
                this.editModeMouseDblclickHandler(e);
            else
                this.viewModeMouseDblclickHandler(e);
            // e.stopPropagation();
        }, false);

        // Handle keyboard 
        window.addEventListener('keydown', (e: KeyboardEvent) => {
            console.log("window.addEventListener keydown ...", e.key);
            if (e.key == 'Shift') {
                this.setMode(this.mode_index + 1);
                return;
            }

            if (this.isEditMode())
                this.editModeKeyDownHandler(e);
            else
                this.viewModeKeyDownHandler(e);
            // e.stopPropagation();
        }, false);

        window.addEventListener('keyup', (e: KeyboardEvent) => {
            console.log("window.addEventListener keyup ...", e.key);
            if (this.isEditMode())
                this.editModeKeyUpHandler(e);
            else
                this.viewModeKeyUpHandler(e);
            // e.stopPropagation();
        }, false);
    }

    setZoom(index: number) {
        // Bad case: (-1 % ZOOM_LEVELS.length) == -1
        if (index < 0) {
            index = ZOOM_LEVELS.length;
        }
        index = index % ZOOM_LEVELS.length;
        this.zoom_index = index;
        this.brush.scale(ZOOM_LEVELS[this.zoom_index], ZOOM_LEVELS[this.zoom_index]);
        this.redraw();

        console.log("Set Zoom: index = ", index, "scale: ", ZOOM_LEVELS[index]);
    }

    redraw() {
        this.brush.clearRect(0, 0, this.canvas.width, this.canvas.height);
        // Draw image ...

        // Draw regions ...
        for (let i = 0; i < this.regions.length; i++) {
            if (i === this.selected_index)
                this.regions[i].draw(this.brush, true);
            else
                this.regions[i].draw(this.brush, false);
        }

        // Draw drawing polygon ...
        // if (this.drawing_shape === ShapeID.Polygon) {
        //     this.drawing_polygon.draw(this.brush, false);
        // }


        let r = new Rectangle(new Point(10, 10), new Point(200, 200));
        r.draw(this.brush, true);
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

    // Find shape object
    private findInside(p: Point): [number, number] {
        for (let i = 0; i < this.regions.length; i++) {
            if (this.regions[i].bodyHas(p))
                return [i, 0];
        }
        return [-1, 0];
    }

    private findVertex(p: Point): [number, number] {
        let vertex_index = -1;
        for (let i = 0; i < this.regions.length; i++) {
            vertex_index = this.regions[i].whichVertexHas(p);
            if (vertex_index >= 0)
                return [i, vertex_index];
        }
        return [-1, -1];
    }

    private findEdge(p: Point): [number, number] {
        let edge_index = -1;
        for (let i = 0; i < this.regions.length; i++) {
            edge_index = this.regions[i].whichEdgeHas(p);
            if (edge_index >= 0)
                return [i, edge_index];
        }
        return [-1, -1];
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

