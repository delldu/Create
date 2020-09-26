// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

const DISTANCE_THRESHOLD = 2;
const EDGE_LINE_WIDTH = 1;
const VERTEX_COLOR = "#ff0000";

const ZOOM_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4, 5, 6, 7, 8, 9, 10];
const DEFAULT_ZOOM_LEVEL = 3; // 1.0 index

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

    onLine(p1: Point, p2: Point): boolean {
        if (this.x < Math.min(p1.x, p2.x) - DISTANCE_THRESHOLD || this.x > Math.max(p1.x, p2.x) + DISTANCE_THRESHOLD)
            return false;
        if (this.y < Math.min(p1.y, p2.y) - DISTANCE_THRESHOLD || this.y > Math.max(p1.y, p2.y) + DISTANCE_THRESHOLD)
            return false;
        if (p1.x == p2.x)
            return true;
        let y = (p2.y - p1.y) / (p2.x - p1.x) * (this.x - p1.x) + p1.y; // y = (y2 - y1)/(x2 - x1) * (x - x1) + y1;
        return Math.abs(y - this.y) <= DISTANCE_THRESHOLD;
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

    clone(): Box {
        let c = new Box(this.x, this.y, this.w, this.h);
        return c;
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

    isclick(): boolean {
        let d = this.start.distance(this.stop);
        return d <= DISTANCE_THRESHOLD;
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

    move(deltaX: number, deltaY: number) {
        for (let p of this.points) {
            p.x += deltaX;
            p.y += deltaY;
        }
    }

    // Bounding Box
    bbox(): Box {
        let n = this.points.length;
        if (n < 1)
            return new Box(0, 0, 0, 0);
        let x1 = this.points[0].x;
        let y1 = this.points[0].x;
        let x2 = this.points[0].y;
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
            if (p.y < Math.min(p1.y, p2.y) - DISTANCE_THRESHOLD)
                continue;
            if (p.y > Math.max(p1.y, p2.y) + DISTANCE_THRESHOLD)
                continue;

            let x = (p.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x;
            if (x > p.x - DISTANCE_THRESHOLD)
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
            if (p.distance(this.points[i]) <= DISTANCE_THRESHOLD)
                return i;
        }
        return -1;
    }

    draw(brush: CanvasRenderingContext2D, selected: boolean) {
        let n = this.points.length;
        if (n < 1)
            return;
        brush.save();
        if (selected) {
            brush.fillStyle = VERTEX_COLOR;
            brush.globalAlpha = 0.5;
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

    push(s: Shape) {
        this.blobs.push(s);
    }

    delete(index: number) {
        this.blobs.slice(index, 1);
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

    clicked(ctrl: boolean, m: Mouse, brush: CanvasRenderingContext2D) {
        // vertex be clicked ?
        let [v_index, v_sub_index] = this.findVertex(m.start);
        if (v_index >= 0 && v_sub_index >= 0) {
            if (ctrl) {
                // Delete this vertex !!!
                this.blobs[v_index].delete(v_sub_index);
                // Delete more ... whole blob ?
                if (this.blobs[v_index].points.length < 3)
                    this.delete(v_index);
                this.draw(brush);
            } else {
                console.log("Do you want to delete vertex ? please press ctrl key and click vertex.")
            }
            return;
        }

        // edge be clicked with ctrl ?
        let [e_index, e_sub_index] = this.findEdge(m.start);
        if (e_index >= 0 && e_sub_index >= 0) {
            // Add vertex
            if (ctrl) {
                this.blobs[e_index].insert(e_sub_index, m.start);
                this.draw(brush);
            } else {
                console.log("Do you want to add vertex ? please press ctrl key and click edge.")
            }
            return;
        }

        // whold blob be clicked ?
        let b_index = this.findBlob(m.start);
        if (b_index >= 0) {
            // Selected/remove selected flag
            if (this.selected_index == b_index) {
                this.selected_index = -1;
                this.blobs[b_index].draw(brush, false);
            } else {
                this.selected_index = b_index;
                this.draw(brush);
            }
            return;
        }
    }

    draging(m: Mouse, brush: CanvasRenderingContext2D) {
        // vertex could be draged ?
        brush.save();

        let box = m.mbbox();
        brush.beginPath();
        brush.moveTo(box.x, box.y);
        brush.lineTo(box.x, box.y + box.h);
        brush.lineTo(box.x + box.w, box.y + box.h);
        brush.lineTo(box.x + box.w, box.y);
        // brush.lineTo(box.x, box.y);
        brush.closePath();
        brush.stroke();

        brush.restore();
    }

    draged(ctrl: boolean, m: Mouse, brush: CanvasRenderingContext2D) {
        // vertex could be draged ? this will change blob
        let [v_index, v_sub_index] = this.findVertex(m.start);
        if (v_index >= 0 && v_sub_index >= 0) {
            if (ctrl) {
                let deltaX = m.stop.x - m.start.x;
                let deltaY = m.stop.y - m.start.y;
                this.blobs[v_index].points[v_sub_index].x += deltaX;
                this.blobs[v_index].points[v_sub_index].y += deltaX;
                this.draw(brush);
            } else {
                console.log("Do you drag vertex ? please press ctrl key and drag vertex.")
            }
            return;
        }

        // whold blob could be draged ?
        let b_index = this.findBlob(m.start);
        if (b_index >= 0) {
            let deltaX = m.stop.x - m.start.x;
            let deltaY = m.stop.y - m.start.y;
            this.blobs[b_index].move(deltaX, deltaY);
            this.draw(brush);
        } else {
            // Add new blob
            let box = m.bbox();
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
            this.push(blob);
            this.draw(brush); // redraw
        } // end of b_index 
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
        this.canvas.tabIndex = -1;  // Support keyboard event

        this.brush = this.canvas.getContext('2d') as CanvasRenderingContext2D;

        this.shape_blobs = new ShapeBlobs();
        this.shape_stack = new ShapeStack();

        // Line width and color
        this.brush.strokeStyle = VERTEX_COLOR;
        this.brush.lineWidth = EDGE_LINE_WIDTH;

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
            // Draging Vertex, Draging blob, Draging blank ...

            this.image_stack.restore(this.brush);

            let rect = this.mouse.mbbox();
            // make sure rect including border
            let erect = rect.clone();
            erect.extend(EDGE_LINE_WIDTH);
            this.image_stack.save(this.brush, erect);

            this.shape_blobs.draging(this.mouse, this.brush);
        } else {
            this.canvas.style.cursor = "default";
        }
    }

    private editModeMouseUpHandler(e: MouseEvent) {
        // Clicking
        if (this.mouse.isclick()) {
            console.log("Click ..., which click source ? ...", e);
            this.shape_blobs.clicked(e.ctrlKey, this.mouse, this.brush);
            return;
        }

        // Draging whole blob? vertex ? blank ?
        if (!this.mouse.isclick() && this.mouse.pressed) {
            this.shape_blobs.draged(e.ctrlKey, this.mouse, this.brush);
            return;
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
        // if (e.key === "d") {
        //     delete selected objects ..., redraw ...
        //     return;
        // }

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

            // this.mouse.draged = true;

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
        if (! this.background_loaded) {
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