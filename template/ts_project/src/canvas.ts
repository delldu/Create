// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

const BRUSH_LINE_WIDTH = 1;
const BRUSH_LINE_COLOR = "#ff0000";
const BRUSH_FILL_COLOR = "#ff00ff";
const MOUSE_DISTANCE_THRESHOLD = 2;

const ZOOM_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4, 5, 6, 7, 8, 9, 10];
const DEFAULT_ZOOM_LEVEL = 3; // 1.0 index

class Point {
    x: number;
    y: number;

    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }

    clone(): Point {
        let b = new Point(this.x, this.y);
        return b;
    }

    offset(deltaX: number, deltaY: number) {
        this.x += deltaX;
        this.y += deltaY;
    }

    zoom(s: number) {
        this.x *= s;
        this.y *= s;
    }

    // euclid distance
    distance(p: Point): number {
        return Math.sqrt((p.x - this.x) * (p.x - this.x) + (p.y - this.y) * (p.y - this.y));
    }

    onLine(p1: Point, p2: Point): boolean {
        if (this.x < Math.min(p1.x, p2.x) - MOUSE_DISTANCE_THRESHOLD || this.x > Math.max(p1.x, p2.x) + MOUSE_DISTANCE_THRESHOLD)
            return false;
        if (this.y < Math.min(p1.y, p2.y) - MOUSE_DISTANCE_THRESHOLD || this.y > Math.max(p1.y, p2.y) + MOUSE_DISTANCE_THRESHOLD)
            return false;
        if (p1.x == p2.x)
            return true;
        let y = (p2.y - p1.y) / (p2.x - p1.x) * (this.x - p1.x) + p1.y; // y = (y2 - y1)/(x2 - x1) * (x - x1) + y1;
        return Math.abs(y - this.y) <= MOUSE_DISTANCE_THRESHOLD;
    }
}

class Box {
    x: number;
    y: number;
    w: number;
    h: number;

    constructor(x: number, y: number, w: number, h: number) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
    }

    valid():boolean {
        if (this.x < 0 || this.y < 0) {
            console.log("Start position is not valid. give up.");
            return false;
        }
        if (this.w < BRUSH_LINE_WIDTH * 8 || this.h < BRUSH_LINE_WIDTH * 8) {
            console.log("Too small blob, give up.");
            return false;
        }
        return true;
    }

    clone(): Box {
        let c = new Box(this.x, this.y, this.w, this.h);
        return c;
    }

    offset(deltaX: number, deltaY: number) {
        this.x += deltaX;
        this.y += deltaY;
    }

    zoom(s: number) {
        this.x *= s;
        this.y *= s;
        this.w *= s;
        this.h *= s;
    }

    extend(delta: number) {
        this.x = this.x - delta;
        this.y = this.y - delta;
        if (this.x < 0)
            this.x = 0;
        if (this.y < 0)
            this.y = 0;
        this.w += 2 * delta;
        this.h += 2 * delta;
    }

    // is there intersect between this and box ?
    intersect(box: Box): boolean {
        // box1 == this, box2 == box
        if (this.x + this.w < box.x || this.x > box.x + box.w)
            return false;
        if (this.y + this.h < box.y || this.y > box.y + box.h)
            return false;
        return true;
    }
}

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

// AMD Mode -- A: add, M: move, D: delete
class Shape {
    label: string;
    points: Array < Point > ;

    constructor() {
        this.label = "";
        this.points = new Array < Point > ();
    }

    clone(): Shape {
        let c = new Shape();
        c.label = this.label;
        for (let p of this.points)
            c.push(p);
        return c;
    }

    offset(deltaX: number, deltaY: number) {
        for (let p of this.points)
            p.offset(deltaX, deltaY);
    }

    zoom(s: number) {
        for (let p of this.points)
            p.zoom(s);
    }

    push(p: Point) {
        this.points.push(new Point(p.x, p.y)); // Dot share, or will be disater !!!
    }

    insert(index: number, p: Point) {
        this.points.splice(index, 0, new Point(p.x, p.y));
    }

    delete(index: number) {
        this.points.splice(index, 1);
    }

    pop() {
        this.points.pop();
    }

    // Bounding Box
    bbox(): Box {
        let n = this.points.length;
        if (n < 1)
            return new Box(0, 0, 0, 0);
        let x1 = this.points[0].x;
        let y1 = this.points[0].y;
        let x2 = this.points[0].x;
        let y2 = this.points[0].y;
        for (let i = 1; i < n; i++) {
            if (x1 > this.points[i].x)
                x1 = this.points[i].x;
            if (x2 < this.points[i].x)
                x2 = this.points[i].x;
            if (y1 > this.points[i].y)
                y1 = this.points[i].y;
            if (y2 < this.points[i].y)
                y2 = this.points[i].y;
        }
        let box = new Box(x1, y1, x2 - x1, y2 - y1);
        return box;
    }

    // p is inside of polygon ?
    inside(p: Point): boolean {
        let cross = 0;
        let n = this.points.length;
        if (n < 3)
            return false;

        for (let i = 0; i < n; i++) {
            let p1 = this.points[i];
            let p2 = this.points[(i + 1) % n];

            if (p1.y == p2.y)
                continue;
            if (p.y < Math.min(p1.y, p2.y) - MOUSE_DISTANCE_THRESHOLD)
                continue;
            if (p.y > Math.max(p1.y, p2.y) + MOUSE_DISTANCE_THRESHOLD)
                continue;

            let x = (p.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x;
            if (x > p.x - MOUSE_DISTANCE_THRESHOLD)
                cross++;
        }
        return (cross % 2 == 1);
    }

    // p is on edge of polygon ?
    onEdge(p: Point): number {
        let n = this.points.length;
        if (n < 3)
            return -1;
        for (let i = 0; i < n; i++) {
            if (p.onLine(this.points[i], this.points[(i + 1) % n]))
                return i;
        }
        return -1;
    }

    // p is on vertex of polygon ?
    onVertex(p: Point): number {
        let n = this.points.length;
        for (let i = 0; i < n; i++) {
            if (p.distance(this.points[i]) <= MOUSE_DISTANCE_THRESHOLD)
                return i;
        }
        return -1;
    }

    // dump vertex for debug
    dump() {
        console.log("Dump vertex ...");
        let n = this.points.length;
        for (let i = 0; i < n; i++) {
            console.log("  i:", i, this.points[i]);
        }
    }

    draw(brush: CanvasRenderingContext2D, selected: boolean) {
        let n = this.points.length;
        if (n < 1)
            return;
        brush.save();
        if (selected) {
            brush.fillStyle = BRUSH_FILL_COLOR;
            brush.globalAlpha = 0.5;
        } else {
            brush.strokeStyle = BRUSH_LINE_COLOR;
            brush.lineWidth = BRUSH_LINE_WIDTH;
        }
        brush.beginPath();
        brush.moveTo(this.points[0].x, this.points[0].y);
        for (let i = 1; i < n; ++i)
            brush.lineTo(this.points[i].x, this.points[i].y);
        // brush.lineTo(this.points[0].x, this.points[0].y); // close loop
        brush.closePath();
        if (selected) {
            brush.fill();
            // this.drawVertex(brush);
        } else {
            brush.stroke();
        }

        // Draw vertex
        brush.fillStyle = BRUSH_FILL_COLOR;
        brush.globalAlpha = 1.0;
        for (let p of this.points) {
            brush.beginPath();
            brush.arc(p.x, p.y, MOUSE_DISTANCE_THRESHOLD, 0, 2 * Math.PI, false);
            brush.closePath();
            brush.fill();
        }
        brush.restore();
    }

    dragingDraw(brush: CanvasRenderingContext2D) {
        let n = this.points.length;
        if (n < 1)
            return;

        brush.save();
        brush.strokeStyle = BRUSH_LINE_COLOR;
        brush.lineWidth = BRUSH_LINE_WIDTH;
        brush.setLineDash([1, 1]);

        brush.beginPath();
        brush.moveTo(this.points[0].x, this.points[0].y);
        for (let i = 1; i < n; ++i)
            brush.lineTo(this.points[i].x, this.points[i].y);
        // brush.lineTo(this.points[0].x, this.points[0].y); // close loop
        brush.closePath();
        brush.stroke();
        brush.restore();
    }
}

class ShapeBlobs {
    blobs: Array < Shape > ;
    selected_index: number;

    constructor() {
        this.blobs = new Array < Shape > ();
        this.selected_index = -1;
    }

    clone(): ShapeBlobs {
        let c = new ShapeBlobs();
        c.selected_index = this.selected_index;
        for (let i = 0; i < this.blobs.length; i++)
            c.push(this.blobs[i]);
        return c;
    }

    offset(deltaX: number, deltaY: number) {
        for (let b of this.blobs)
            b.offset(deltaX, deltaY);
    }

    zoom(s: number) {
        for (let b of this.blobs)
            b.zoom(s);
    }

    push(s: Shape) {
        this.blobs.push(s);
    }

    delete(index: number) {
        this.blobs.splice(index, 1);
    }

    pop() {
        this.blobs.pop();
    }

    draw(brush: CanvasRenderingContext2D) {
        for (let i = 0; i < this.blobs.length; i++) {
            if (i === this.selected_index)
                this.blobs[i].draw(brush, true);
            else
                this.blobs[i].draw(brush, false);
        }
    }

    // Region draw speed up drawing ...
    regionDraw(brush: CanvasRenderingContext2D, rect: Box) {
        for (let i = 0; i < this.blobs.length; i++) {
            let bbox = this.blobs[i].bbox();
            if (!rect.intersect(bbox))
                continue;
            if (i === this.selected_index)
                this.blobs[i].draw(brush, true);
            else
                this.blobs[i].draw(brush, false);
        }
    }

    // find blob via point ...
    findBlob(p: Point): number {
        for (let i = 0; i < this.blobs.length; i++) {
            if (this.blobs[i].inside(p))
                return i;
        }
        return -1;
    }

    // find blob, vertex
    findVertex(p: Point): [number, number] {
        for (let i = 0; i < this.blobs.length; i++) {
            let vi = this.blobs[i].onVertex(p);
            if (vi >= 0)
                return [i, vi];
        }
        return [-1, -1];
    }

    // find edge via point
    findEdge(p: Point): [number, number] {
        for (let i = 0; i < this.blobs.length; i++) {
            let ei = this.blobs[i].onEdge(p);
            if (ei >= 0)
                return [i, ei];
        }
        return [-1, -1];
    }

    // if return true -- need redraw, else skip
    clicked(ctrl: boolean, m: Mouse, brush: CanvasRenderingContext2D): boolean {
        // vertex be clicked ?
        let [v_index, v_sub_index] = this.findVertex(m.start);
        if (v_index >= 0 && v_sub_index >= 0) {
            if (ctrl) {
                // Delete this vertex !!!
                this.blobs[v_index].delete(v_sub_index);
                // Delete more ... whole blob ?
                if (this.blobs[v_index].points.length < 3)
                    this.delete(v_index);
                return true;
            } else {
                console.log("Do you want to delete vertex ? please press ctrl key and click vertex.")
            }
            return false;
        }

        // edge be clicked with ctrl ?
        let [e_index, e_sub_index] = this.findEdge(m.start);
        if (e_index >= 0 && e_sub_index >= 0) {
            // Add vertex
            if (ctrl) {
                this.blobs[e_index].insert(e_sub_index + 1, m.start);
                return true;
            } else {
                console.log("Do you want to add vertex ? please press ctrl key and click edge.")
            }
            return false;
        }

        // whold blob be clicked ?
        let b_index = this.findBlob(m.start);
        if (b_index >= 0) {
            // Selected/remove selected flag
            if (this.selected_index == b_index) {
                this.selected_index = -1;
            } else {
                this.selected_index = b_index;
            }
            return true;
        }
        return false;
    }

    // no need any redraw for using save/restore methods
    draging(ctrl: boolean, m: Mouse, brush: CanvasRenderingContext2D, image_stack: ImageStack) {
        // vertex could be draged ?

        // draging vertex ?
        let [v_index, v_sub_index] = this.findVertex(m.start);
        if (v_index >= 0 && v_sub_index >= 0) {
            let n = this.blobs[v_index].points.length;
            let blob = new Shape();
            blob.push(this.blobs[v_index].points[(v_sub_index - 1) % n]);
            blob.push(this.blobs[v_index].points[(v_sub_index + 1) % n]);
            blob.push(m.moving);

            let rect = blob.bbox();
            if (! rect.valid())
                return;

            image_stack.restore(brush);
            rect.extend(BRUSH_LINE_WIDTH);
            image_stack.save(brush, rect);

            blob.dragingDraw(brush);

            return;
        }

        // draging whold blob ?
        let b_index = this.findBlob(m.start);
        if (b_index >= 0) {
            let deltaX = m.moving.x - m.start.x;
            let deltaY = m.moving.y - m.start.y;
            let blob = this.blobs[b_index].clone();
            blob.offset(deltaX, deltaY);

            let rect = blob.bbox();
            if (! rect.valid())
                return;

            image_stack.restore(brush);
            rect.extend(BRUSH_LINE_WIDTH);
            image_stack.save(brush, rect);

            blob.dragingDraw(brush);
            return;
        } else {
            // Add new blob
            let box = m.mbbox();
            if (! box.valid())
                return;

            let blob = new Shape();
            if (ctrl) { // Add 3x3 rectangle, heavy blob
                blob.push(new Point(box.x, box.y));
                blob.push(new Point(box.x, box.y + box.h / 2));
                blob.push(new Point(box.x, box.y + box.h));
                blob.push(new Point(box.x + box.w / 2, box.y + box.h));
                blob.push(new Point(box.x + box.w, box.y + box.h));
                blob.push(new Point(box.x + box.w, box.y + box.h / 2));
                blob.push(new Point(box.x + box.w, box.y));
                blob.push(new Point(box.x + box.w / 2, box.y));
            } else { // Add 2x2 rectangle, light blob
                blob.push(new Point(box.x, box.y));
                blob.push(new Point(box.x, box.y + box.h));
                blob.push(new Point(box.x + box.w, box.y + box.h));
                blob.push(new Point(box.x + box.w, box.y));
            }

            let rect = blob.bbox();
            if (! rect.valid())
                return;

            image_stack.restore(brush);
            rect.extend(BRUSH_LINE_WIDTH);
            image_stack.save(brush, rect);

            blob.dragingDraw(brush);
            return;
        } // end of b_index 
    }

    // return true -- need redraw, else false
    draged(ctrl: boolean, m: Mouse, brush: CanvasRenderingContext2D): boolean {
        // vertex could be draged ? this will change blob
        let [v_index, v_sub_index] = this.findVertex(m.start);
        if (v_index >= 0 && v_sub_index >= 0) {
            if (ctrl) {
                let deltaX = m.stop.x - m.start.x;
                let deltaY = m.stop.y - m.start.y;
                this.blobs[v_index].points[v_sub_index].x += deltaX;
                this.blobs[v_index].points[v_sub_index].y += deltaY;
                return true;
            } else {
                console.log("Do you want draging vertex ? please press ctrl key and drag vertex.")
            }
            return false;
        }

        // whold blob could be draged ?
        let b_index = this.findBlob(m.start);
        if (b_index >= 0) {
            let deltaX = m.stop.x - m.start.x;
            let deltaY = m.stop.y - m.start.y;
            this.blobs[b_index].offset(deltaX, deltaY);
        } else {
            // Add new blob
            let box = m.bbox();
            if (! box.valid())
                return false;

            let blob = new Shape();
            if (ctrl) { // Add 3x3 rectangle, heavy blob
                blob.push(new Point(box.x, box.y));
                blob.push(new Point(box.x, box.y + box.h / 2));
                blob.push(new Point(box.x, box.y + box.h));
                blob.push(new Point(box.x + box.w / 2, box.y + box.h));
                blob.push(new Point(box.x + box.w, box.y + box.h));
                blob.push(new Point(box.x + box.w, box.y + box.h / 2));
                blob.push(new Point(box.x + box.w, box.y));
                blob.push(new Point(box.x + box.w / 2, box.y));
            } else { // Add 2x2 rectangle, light blob
                blob.push(new Point(box.x, box.y));
                blob.push(new Point(box.x, box.y + box.h));
                blob.push(new Point(box.x + box.w, box.y + box.h));
                blob.push(new Point(box.x + box.w, box.y));
            }
            box = blob.bbox();
            if (! box.valid())
                return false;

            this.push(blob);
        } // end of b_index 
        return true;
    } // end of draged
}

class ShapeStack {
    stack: Array < Shape > ;

    constructor() {
        this.stack = new Array < Shape > ();
    }

    push(blob: Shape) {
        this.stack.push(blob);
    }

    pop(): Shape {
        return this.stack.pop() as Shape;
    }

    reset() {
        this.stack.length = 0;
    }
}

class Canvas {
    canvas: HTMLCanvasElement; // canvas element
    private brush: CanvasRenderingContext2D;
    private background: HTMLImageElement; // Image;
    private background_loaded: boolean;

    mode_index: number;

    // Shape container
    shape_blobs: ShapeBlobs;
    private shape_stack: ShapeStack;
    private image_stack: ImageStack;

    // Zoom control
    zoom_index: number;

    // Handle mouse
    private mouse: Mouse;

    constructor(id: string) {
        this.canvas = document.getElementById(id) as HTMLCanvasElement;
        this.canvas.tabIndex = -1; // Support keyboard event

        this.brush = this.canvas.getContext('2d') as CanvasRenderingContext2D;

        this.shape_blobs = new ShapeBlobs();
        this.shape_stack = new ShapeStack();

        // Line width and color
        this.brush.strokeStyle = BRUSH_LINE_COLOR;
        this.brush.lineWidth = BRUSH_LINE_WIDTH;

        this.zoom_index = DEFAULT_ZOOM_LEVEL;

        this.mode_index = 0;

        this.mouse = new Mouse();
        this.registerEventHandlers();

        // Create background for canvas
        this.background = document.createElement('img') as HTMLImageElement;
        this.background.id = this.canvas.id + "_background_image";
        this.background.style.display = "none";
        this.canvas.appendChild(this.background);
        // this.background.src = "dog.jpg";
        this.background_loaded = false;

        this.image_stack = new ImageStack();
    }

    private loadingBackground() {
        this.background_loaded = false;
        this.background.onload = () => {
            this.background_loaded = true;
            if (this.canvas.width < this.background.naturalWidth)
                this.canvas.width = this.background.naturalWidth;
            if (this.canvas.height < this.background.naturalHeight)
                this.canvas.height = this.background.naturalHeight;

            this.setZoom(DEFAULT_ZOOM_LEVEL);
        }
    }

    setMessage(message: string) {
        console.log(message);
    }

    setMode(index: number) {
        // Bad case: (-1 % 2) == -1
        if (index < 0)
            index = 1;
        index = index % 2;
        this.mode_index = index;
        console.log("Set mode:", this.mode_index);
    }

    getMode(): number {
        return this.mode_index;
    }

    isEditMode(): boolean {
        return this.mode_index > 0;
    }

    private viewModeMouseDownHandler(e: MouseEvent) {
        // console.log("viewModeMouseDownHandler ...", e);
    }

    private viewModeMouseMoveHandler(e: MouseEvent) {
        if (this.mouse.pressed) {
            this.canvas.style.cursor = "pointer";
            // this.canvas.start = new Point(this.mouse.stop.x, this.mouse.stop.y);
        } else {
            this.canvas.style.cursor = "default";
        }
    }

    private viewModeMouseUpHandler(e: MouseEvent) {
        if (!this.mouse.isclick() && this.mouse.pressed) {
            console.log("draging ..., which src object ? moving background ?");
        }
    }

    private editModeMouseDownHandler(e: MouseEvent) {
        // Start:  On selected vertext, On selected Edge, Inside, Blank area
        // e.ctrlKey, e.altKey -- boolean

        this.image_stack.reset();
    }

    private editModeMouseMoveHandler(e: MouseEvent) {
        if (this.mouse.pressed) {
            this.canvas.style.cursor = "pointer";
            this.shape_blobs.draging(e.ctrlKey, this.mouse, this.brush, this.image_stack);
        } else {
            this.canvas.style.cursor = "default";
        }
    }

    private editModeMouseUpHandler(e: MouseEvent) {
        // Clicking
        if (this.mouse.isclick()) {
            if (this.shape_blobs.clicked(e.ctrlKey, this.mouse, this.brush))
                this.redraw();
            return;
        }

        if (!this.mouse.isclick() && this.mouse.pressed) {
            if (this.shape_blobs.draged(e.ctrlKey, this.mouse, this.brush))
                this.redraw();
            return;
        }
    }

    private viewModeKeyDownHandler(e: KeyboardEvent) {
        // console.log("viewModeKeydownHandler ...");
        e.preventDefault();
    }

    private viewModeKeyUpHandler(e: KeyboardEvent) {
        if (e.key === "+") {
            this.setZoom(this.zoom_index + 1);
        } else if (e.key === "=") {
            this.setZoom(DEFAULT_ZOOM_LEVEL);
        } else if (e.key === "-") {
            this.setZoom(this.zoom_index - 1);
        } else if (e.key === 'F1') { // F1 for help
            this.canvas.style.cursor = "help";
            // todo
        } else {
            // todo
        }
        e.preventDefault();
    }


    private editModeKeyDownHandler(e: KeyboardEvent) {
        console.log("editModeKeyDown ...", e.key);
    }

    private editModeKeyUpHandler(e: KeyboardEvent) {
        console.log("editModeKeyUpHandler ...", e.key);
        if (e.key === "d") {
            if (this.shape_blobs.selected_index >= 0) {
                console.log("We will delete blob:", this.shape_blobs.selected_index);
                console.log("Delete before:", this.shape_blobs);
                this.shape_blobs.delete(this.shape_blobs.selected_index);
                console.log("Delete after:", this.shape_blobs);
                this.shape_blobs.selected_index = -1;
                this.redraw();
            }
            return;
        }

        this.viewModeKeyUpHandler(e);
        e.preventDefault();
    }

    private registerEventHandlers() {
        this.canvas.addEventListener('mousedown', (e: MouseEvent) => {
            this.mouse.start.x = e.offsetX;
            this.mouse.start.y = e.offsetY;
            this.mouse.pressed = true;

            if (this.isEditMode())
                this.editModeMouseDownHandler(e);
            else
                this.viewModeMouseDownHandler(e);
        }, false);
        this.canvas.addEventListener('mouseup', (e: MouseEvent) => {
            this.mouse.stop.x = e.offsetX;
            this.mouse.stop.y = e.offsetY;

            // Every this is calm ...
            this.canvas.style.cursor = "default";
            if (this.isEditMode())
                this.editModeMouseUpHandler(e);
            else
                this.viewModeMouseUpHandler(e);

            this.mouse.pressed = false;
            e.stopPropagation();
        }, false);
        this.canvas.addEventListener('mouseover', (e: MouseEvent) => {
            this.redraw();
        }, false);
        this.canvas.addEventListener('mousemove', (e: MouseEvent) => {
            this.mouse.moving.x = e.offsetX;
            this.mouse.moving.y = e.offsetY;

            if (this.isEditMode())
                this.editModeMouseMoveHandler(e);
            else
                this.viewModeMouseMoveHandler(e);
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

        // Handle keyboard 
        this.canvas.addEventListener('keydown', (e: KeyboardEvent) => {
            console.log("window.addEventListener keydown ...", e);
            if (e.key == 'Shift') {
                this.setMode(this.mode_index + 1);
                return;
            }
            if (this.isEditMode())
                this.editModeKeyDownHandler(e);
            else
                this.viewModeKeyDownHandler(e);

            e.preventDefault();
        }, false);

        this.canvas.addEventListener('keyup', (e: KeyboardEvent) => {
            console.log("window.addEventListener keyup ...", e);
            if (this.isEditMode())
                this.editModeKeyUpHandler(e);
            else
                this.viewModeKeyUpHandler(e);
            e.preventDefault();
        }, false);
    }

    setZoom(index: number) {
        // Bad case: (-1 % ZOOM_LEVELS.length) == -1
        if (index < 0) {
            index = ZOOM_LEVELS.length - 1;
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
        if (!this.background_loaded) {
            this.loadingBackground();
        }

        if (this.background_loaded) {
            this.brush.drawImage(this.background, 0, 0);
        }

        // Draw blobs ...
        this.shape_blobs.draw(this.brush);
    }
}

class ImagePatch {
    rect: Box;
    data: ImageData;

    constructor(rect: Box, data: ImageData) {
        this.rect = rect;
        this.data = data;
    }
}

class ImageStack {
    stack: Array < ImagePatch > ;

    constructor() {
        this.stack = new Array < ImagePatch > ();
    }

    save(src: CanvasRenderingContext2D, rect: Box) {
        let data = src.getImageData(rect.x, rect.y, rect.w, rect.h);
        let patch = new ImagePatch(rect, data);
        this.stack.push(patch);
    }

    restore(dst: CanvasRenderingContext2D) {
        if (this.stack.length < 1) // empty
            return;

        let patch = this.stack.pop();
        if (patch && patch.rect.w > 0 && patch.rect.h > 0)
            dst.putImageData(patch.data, patch.rect.x, patch.rect.y);
    }

    reset() {
        this.stack.length = 0;
    }
}