// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

class Point {
    static THRESHOLD = 2;
    constructor(public x: number, public y: number) {}

    clone(): Point {
        return new Point(this.x, this.y);
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
        if (this.x < Math.min(p1.x, p2.x) - Point.THRESHOLD || this.x > Math.max(p1.x, p2.x) + Point.THRESHOLD)
            return false;
        if (this.y < Math.min(p1.y, p2.y) - Point.THRESHOLD || this.y > Math.max(p1.y, p2.y) + Point.THRESHOLD)
            return false;
        if (p1.x == p2.x)
            return true;
        let y = (p2.y - p1.y) / (p2.x - p1.x) * (this.x - p1.x) + p1.y; // y = (y2 - y1)/(x2 - x1) * (x - x1) + y1;
        return Math.abs(y - this.y) <= Point.THRESHOLD;
    }
}

class Box {
    constructor(public x: number, public y: number, public w: number, public h: number) {}

    size(): number {
        return this.w * this.w;
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

    // Bounding Box for two points
    static bbox(p1: Point, p2: Point): Box {
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
}

// AMD Mode -- A: add, M: move, D: delete
class Polygon {
    static LINE_WIDTH = 1;
    static LINE_COLOR = "#FF0000";
    static FILL_COLOR = "#FF00FF";

    label: number;
    points: Array < Point > ;

    constructor() {
        this.label = 0;
        this.points = new Array < Point > ();
    }

    reset() {
        this.points.length = 0;
    }

    clone(): Polygon {
        let c = new Polygon();
        c.label = this.label;
        for (let p of this.points)
            c.push(new Point(p.x, p.y)); // Clone Point
        return c;
    }

    offset(deltaX: number, deltaY: number) {
        for (let p of this.points)
            p.offset(deltaX, deltaY);
    }

    // set 3x3 polygon via rectangle
    set3x3(box: Box) {
        this.reset();
        this.push(new Point(box.x, box.y));
        this.push(new Point(box.x, box.y + box.h / 2));
        this.push(new Point(box.x, box.y + box.h));
        this.push(new Point(box.x + box.w / 2, box.y + box.h));
        this.push(new Point(box.x + box.w, box.y + box.h));
        this.push(new Point(box.x + box.w, box.y + box.h / 2));
        this.push(new Point(box.x + box.w, box.y));
        this.push(new Point(box.x + box.w / 2, box.y));
    }

    push(p: Point) {
        this.points.push(new Point(p.x, p.y)); // Do not share, or will be disaster !!!
    }

    addPoint(index: number, p: Point) {
        this.points.splice(index, 0, new Point(p.x, p.y));
    }

    delPoint(index: number) {
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
        return new Box(x1, y1, x2 - x1, y2 - y1);
    }

    size(): number {
        return this.bbox().size();
    }

    // here j is a JSON Object
    loadJSON(j: any) {
        let blob = new Polygon();
        this.reset();
        for (let key in j) {
            if (!j.hasOwnProperty(key))
                continue;
            // key == "label" || "points" ...
            if (key == "label") {
                this.label = parseInt(j[key]);
                continue;
            }
            if (key != "points")
                continue;
            // now key === points
            for (let k4 in j[key]) {
                if (!j[key].hasOwnProperty(k4))
                    continue;
                this.push(new Point(j[key][k4].x, j[key][k4].y));
            }
        }
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
            if (p.y < Math.min(p1.y, p2.y) - Point.THRESHOLD)
                continue;
            if (p.y > Math.max(p1.y, p2.y) + Point.THRESHOLD)
                continue;

            let x = (p.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x;
            if (x > p.x - Point.THRESHOLD)
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

    // p is on vertex ?
    onVertex(p: Point): number {
        let n = this.points.length;
        for (let i = 0; i < n; i++) {
            if (p.distance(this.points[i]) <= 2 * Point.THRESHOLD)
                return i;
        }
        return -1;
    }

    setPath(brush: CanvasRenderingContext2D) {
        let n = this.points.length;
        if (n < 1)
            return;
        brush.translate(0.5, 0.5);
        brush.beginPath();
        brush.moveTo(this.points[0].x, this.points[0].y);
        for (let i = 1; i < n; ++i)
            brush.lineTo(this.points[i].x, this.points[i].y);
        brush.closePath();
    }

    draw(brush: CanvasRenderingContext2D, selected: boolean) {
        if (selected) {
            this.fillRegion(brush);
        } else {
            this.drawBorder(brush);
        }
        this.fillVertex(brush);
    }

    drawBorder(brush: CanvasRenderingContext2D) {
        brush.save();
        brush.strokeStyle = Polygon.LINE_COLOR;
        brush.lineWidth = Polygon.LINE_WIDTH;
        this.setPath(brush);
        brush.stroke();
        brush.restore();
    }

    drawDashBorder(brush: CanvasRenderingContext2D) {
        brush.save();
        brush.strokeStyle = Polygon.LINE_COLOR;
        brush.lineWidth = Polygon.LINE_WIDTH;
        brush.setLineDash([1, 1]);
        this.setPath(brush);
        brush.stroke();
        brush.restore();
    }

    drawShiftBorder(brush: CanvasRenderingContext2D) {
        let n = this.points.length;
        if (n < 1)
            return;
        this.fillVertex(brush);
        brush.save();
        brush.strokeStyle = Polygon.LINE_COLOR;
        brush.lineWidth = Polygon.LINE_WIDTH;
        brush.setLineDash([1, 1]);
        brush.translate(0.5, 0.5);
        brush.moveTo(this.points[0].x, this.points[0].y);
        for (let i = 1; i < n; ++i)
            brush.lineTo(this.points[i].x, this.points[i].y);
        brush.stroke();
        brush.restore();
    }

    fillRegion(brush: CanvasRenderingContext2D) {
        brush.save();
        brush.fillStyle = Polygon.FILL_COLOR;
        brush.globalAlpha = 0.25;
        this.setPath(brush);
        brush.fill();
        brush.restore();
    }

    fillVertex(brush: CanvasRenderingContext2D) {
        brush.save();
        brush.fillStyle = Polygon.FILL_COLOR;
        brush.globalAlpha = 1.0;
        brush.translate(0.5, 0.5);
        for (let p of this.points) {
            brush.beginPath();
            brush.arc(p.x, p.y, Point.THRESHOLD, 0, 2 * Math.PI, false);
            brush.closePath();
            brush.fill();
        }
        brush.restore();
    }
}

class Shape {
    blobs: Array < Polygon > ;
    selected_index: number;

    constructor() {
        this.blobs = new Array < Polygon > ();
        this.selected_index = -1;
    }

    reset() {
        this.blobs.length = 0;
        this.selected_index = -1;
    }

    // here s is a JSON string
    loadJSON(s: string) {
        try {
            let j = JSON.parse(s);
            this.blobs.length = 0; // reset
            for (let k1 in j) {
                if (!j.hasOwnProperty(k1))
                    continue;
                if (k1 == "selected_index") {
                    this.selected_index = parseInt(j[k1]);
                    continue;
                }
                if (k1 != "blobs")
                    continue;
                // now k1 === "blobs"
                for (let k2 in j[k1]) {
                    if (!j[k1].hasOwnProperty(k2))
                        continue;
                    // Here k2 is number 0, 1 ...
                    let blob = new Polygon();
                    blob.loadJSON(j[k1][k2]);
                    this.push(blob);
                } // end of k2
            }
        } catch {
            console.log("JSON Parse error.");
            return;
        }
    }

    push(s: Polygon) {
        this.blobs.push(s.clone());
    }

    addBlob(index: number, s: Polygon) {
        this.blobs.splice(index, 0, s.clone());
    }

    delBlob(index: number) {
        this.blobs.splice(index, 1);
    }

    pop() {
        this.blobs.pop();
    }

    draw(brush: CanvasRenderingContext2D) {
        for (let i = 0; i < this.blobs.length; i++)
            this.blobs[i].draw(brush, (i == this.selected_index));
    }

    // find blob via point ...
    findBlob(p: Point): number {
        let index = -1;
        let smallest = 8196 * 8196; // big size
        for (let i = 0; i < this.blobs.length; i++) {
            if (!this.blobs[i].inside(p))
                continue;
            // Find smallest
            let size = this.blobs[i].size();
            if (smallest >= size) {
                index = i;
                smallest = size;
            }
        }
        return index;
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

    // Edit Vertex/Delete
    vertextClickOver(from: Point): boolean {
        // if hit vertex, delete it
        let [v_index, v_sub_index] = this.findVertex(from);
        if (v_index >= 0 && v_sub_index >= 0) {
            // Delete vertex
            this.blobs[v_index].delPoint(v_sub_index);
            // Delete more ... whole blob ?
            if (this.blobs[v_index].points.length < 3)
                this.delBlob(v_index);
            return true;
        }
        return false;
    }
    
    // Edit Vertex/Add
    edgeClickOver(from: Point): boolean {
        // if hit edge, add one vertext
        let [e_index, e_sub_index] = this.findEdge(from);
        if (e_index >= 0 && e_sub_index >= 0) {
            this.blobs[e_index].addPoint(e_sub_index + 1, from);
            return true;
        }
        return false;
    }

    // Select Blob
    blobClickOver(from: Point): boolean {
        // if hit it, select/remove selected flag
        let b_index = this.findBlob(from);
        if (b_index >= 0) {
            if (this.selected_index == b_index) {
                this.selected_index = -1;
            } else {
                this.selected_index = b_index;
            }
            return true;
        }
        return false;
    }

    // Edit Vertext/Change Position
    vertextDragging(from: Point, to: Point, target: Polygon): boolean {
        let [v_index, v_sub_index] = this.findVertex(from);
        if (v_index >= 0 && v_sub_index >= 0) {
            let n = this.blobs[v_index].points.length;
            // Bad case -1 % n = -1 !!!
            // So we use (n + v_sub_index - 1)%n instead of (v_sub_index - 1) % n
            target.reset();
            target.push(this.blobs[v_index].points[(n + v_sub_index - 1) % n]);
            target.push(this.blobs[v_index].points[(v_sub_index + 1) % n]);
            target.push(to);
            return true;
        }
        return false;
    }

    // Edit Blob/Change Position
    blobDragging(from: Point, to: Point, target: Polygon): boolean {
        // dragging exist blob ?
        let b_index = this.findBlob(from);
        if (b_index >= 0) {
            let deltaX = to.x - from.x;
            let deltaY = to.y - from.y;

            target.reset(); // Clone
            for (let i = 0; i < this.blobs[b_index].points.length; i++)
                target.push(this.blobs[b_index].points[i].clone());

            target.offset(to.x - from.x, to.y - from.y);
            return true;
        }
        return false;
    }

    // Create New Blob
    newBlobDragging(from: Point, to: Point, target: Polygon): boolean {
        // dragging new blob (created by box)
        target.set3x3(Box.bbox(from, to));
        return true;
    }

    // Edit Blob/Change Position
    vertextDragOver(from: Point, to: Point): boolean {
        // vertex could be dragged ? yes will change blob
        let [v_index, v_sub_index] = this.findVertex(from);
        if (v_index >= 0 && v_sub_index >= 0) {
            this.blobs[v_index].points[v_sub_index].offset(to.x - from.x, to.y - from.y);
            return true;
        }
        return false;
    }

    // Edit Blob/Change Position
    blobDragOver(from: Point, to: Point): boolean {
        let b_index = this.findBlob(from);
        if (b_index >= 0) {
            this.blobs[b_index].offset(to.x - from.x, to.y - from.y);
            return true;
        }
        return false;
    }

    // Create New Blob
    newBlobDragOver(from: Point, to: Point): boolean {
        let box = Box.bbox(from, to);
        let blob = new Polygon();
        blob.set3x3(box);
        this.push(blob);
        return true;
    }
}

class Canvas {
    static ZOOM_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4, 5, 6, 7, 8, 9, 10];
    static DEFAULT_ZOOM_LEVEL = 3; // 1.0 index

    canvas: HTMLCanvasElement; // canvas element
    private brush: CanvasRenderingContext2D;

    background: HTMLImageElement; // Image;
    private background_loaded: boolean;
    shape: Shape; // shape

    private mode_index: number;
    // Zoom control
    private zoom_index: number;

    private shift_blob: Polygon;
    private image_stack: ImageStack;

    // Handle mouse, keyboard
    private mouse: Mouse;
    private keyboard: Keyboard;
    key: string;

    constructor(id: string) {
        this.canvas = document.getElementById(id) as HTMLCanvasElement;
        this.canvas.tabIndex = -1; // Support keyboard event

        this.brush = this.canvas.getContext('2d') as CanvasRenderingContext2D;

        this.shape = new Shape();
        this.shift_blob = new Polygon();

        // Default brush line width, color
        this.brush.strokeStyle = Polygon.LINE_COLOR;
        this.brush.lineWidth = Polygon.LINE_WIDTH;
        this.brush.fillStyle = Polygon.FILL_COLOR;

        this.zoom_index = Canvas.DEFAULT_ZOOM_LEVEL;

        this.mode_index = 0;

        this.mouse = new Mouse();
        this.keyboard = new Keyboard();

        this.registerEventHandlers();

        this.background = new Image() as HTMLImageElement;
        this.background_loaded = false;

        this.image_stack = new ImageStack();

        this.key = "";
    }

    reset() {
        this.shape.reset();
        this.shift_blob.reset();
        this.zoom_index = Canvas.DEFAULT_ZOOM_LEVEL;
        this.mode_index = 0;
        this.image_stack.reset();
        this.background_loaded = false;
        this.key = "";
    }

    loadBackground(dataurl: string) {
        this.background_loaded = false;
        this.background.src = dataurl;
        this.canvas.width = this.background.naturalWidth;
        this.canvas.height = this.background.naturalHeight;
        this.setZoom(Canvas.DEFAULT_ZOOM_LEVEL);
        this.background_loaded = true;
    }

    loadBlobs(blobs: string) {
        this.shape.loadJSON(blobs);
    }

    saveBlobs(): string {
        return JSON.stringify(this.shape, undefined, 2);
    }

    getMode(): number {
        return this.mode_index;
    }

    setMode(index: number) {
        // Bad case: (-1 % 2) == -1
        if (index < 0)
            index = 1;
        index = index % 2;
        this.mode_index = index;
        console.log("Canvas: set mode:", this.mode_index);
    }

    isEditMode(): boolean {
        return this.mode_index > 0;
    }

    // EditMode has 4 sub mode:
    // 1. Ctrl mode -- control key pressed: edit polygon, add 3x3, add vertex, dete vertex, drag ...
    // 2. Shift mode -- shift key pressed: free click for polygon ...
    // 3. Alt mode -- alt key pressed: AI create polygon ...
    // 4. Normal mode -- others: create 2x2, select or not, drag/delete selected ...
    // Generally keyboard.mode() return current mode: KeyboardMode.CtrlKeydown, ...

    ctrlModeStart() {
    }

    ctrlModeMouseMoveHandle() {
        if (!this.mouse.pressed()) // Dragging status
            return;

        let t = new Polygon();

        if (this.shape.vertextDragging(this.mouse.start, this.mouse.moving, t)) {
            this.fastDrawMovingObject(t);
            return;
        }

        if (this.shape.blobDragging(this.mouse.start, this.mouse.moving, t)) {
            this.fastDrawMovingObject(t);
            return;
        }

        if (this.shape.newBlobDragging(this.mouse.start, this.mouse.moving, t)) {
            this.fastDrawMovingObject(t);
            return;
        }
    }

    ctrlModeMouseUpHandle() {
        if (this.mouse.overStatus() == MouseOverStatus.ClickOver) {
            if (this.shape.vertextClickOver(this.mouse.start)) {
                this.redraw();
                return;
            }
            if (this.shape.edgeClickOver(this.mouse.start)) {
                this.redraw();
                return;
            }
        } else if (this.mouse.overStatus() == MouseOverStatus.DragOver) {
            if (this.shape.vertextDragOver(this.mouse.start, this.mouse.stop)) {
                this.redraw();
                return;
            }

            if (this.shape.blobDragOver(this.mouse.start, this.mouse.stop)) {
                this.redraw();
                return;
            }

            if (this.shape.newBlobDragOver(this.mouse.start, this.mouse.stop)) {
                this.redraw();
                return;
            }
        }
    }

    ctrlModeStop() {
        this.keyboard.pop();
    }

    shiftModeStart() {
        this.shift_blob.reset();
    }

    shiftModeMouseMoveHandle() {
        let box = this.shift_blob.bbox();
        box.extend(Polygon.LINE_WIDTH * 2);
        this.image_stack.restore(this.brush);
        this.image_stack.save(this.brush, box);
        this.shift_blob.drawShiftBorder(this.brush);
    }

    shiftModeMouseUpHandle() {
        this.shift_blob.push(this.mouse.stop);
    }

    shiftModeStop() {
        if (this.shift_blob.points.length >= 3) {
            this.shape.push(this.shift_blob);
            this.redraw();
        }
        this.keyboard.pop();
    }

    altModeStart() {
    }

    altModeMouseMoveHandle() {
    }

    altModeMouseUpHandle() {
    }

    altModeStop() {
        this.keyboard.pop();
    }

    normalModeMouseMoveHandle() {
        if (!this.mouse.pressed()) // Dragging status
            return;

        let t = new Polygon();
        if (this.shape.blobDragging(this.mouse.start, this.mouse.moving, t)) {
            this.fastDrawMovingObject(t);
            return;
        }
        if (this.shape.newBlobDragging(this.mouse.start, this.mouse.moving, t)) {
            this.fastDrawMovingObject(t);
            return;
        }
    }

    normalModeMouseUpHandle() {
        if (this.mouse.overStatus() == MouseOverStatus.ClickOver) {
            if (this.shape.blobClickOver(this.mouse.start)) {
                this.redraw(); // NO Fast redraw method, so redraw
                return;
            }
        } else if (this.mouse.overStatus() == MouseOverStatus.DragOver) {
            if (this.shape.blobDragOver(this.mouse.start, this.mouse.stop)) {
                this.redraw();
                return;
            }
            if (this.shape.newBlobDragOver(this.mouse.start, this.mouse.stop)) {
                this.redraw();
                return;
            }
        }
    }

    private viewModeMouseDownHandler(e: MouseEvent) {
        // console.log("viewModeMouseDownHandler ...", e);
    }

    private viewModeMouseMoveHandler(e: MouseEvent) {
        this.canvas.style.cursor = this.mouse.pressed()? "pointer" : "default";
    }

    private viewModeMouseUpHandler(e: MouseEvent) {
        if (this.mouse.overStatus() == MouseOverStatus.DragOver) {
        }
    }

    private editModeMouseDownHandler(e: MouseEvent) {
        this.image_stack.reset();
    }

    private editModeMouseMoveHandler(e: MouseEvent) {
        this.canvas.style.cursor = this.mouse.pressed() ? "pointer" : "default";

        if (this.keyboard.mode() == KeyboardMode.CtrlKeydown) {
            this.ctrlModeMouseMoveHandle();
        } else if (this.keyboard.mode() == KeyboardMode.ShiftKeydown) {
            this.shiftModeMouseMoveHandle();
        } else if (this.keyboard.mode() == KeyboardMode.AltKeydown) {
            this.altModeMouseMoveHandle();
        } else { // Normal mode
            this.normalModeMouseMoveHandle();
        }
    }

    private editModeMouseUpHandler(e: MouseEvent) {
        if (this.keyboard.mode() == KeyboardMode.CtrlKeydown) {
            this.ctrlModeMouseUpHandle();
        } else if (this.keyboard.mode() == KeyboardMode.ShiftKeydown) {
            this.shiftModeMouseUpHandle();
        } else if (this.keyboard.mode() == KeyboardMode.AltKeydown) {
            this.altModeMouseUpHandle();
        } else { // Normal mode
            this.normalModeMouseUpHandle();
        }
    }

    private viewModeKeyDownHandler(e: KeyboardEvent) {
        e.preventDefault();
    }

    private viewModeKeyUpHandler(e: KeyboardEvent) {
        if (e.key === "+") {
            this.setZoom(this.zoom_index + 1);
        } else if (e.key === "=") {
            this.setZoom(Canvas.DEFAULT_ZOOM_LEVEL);
        } else if (e.key === "-") {
            this.setZoom(this.zoom_index - 1);
        } else if (e.key === 'F1') { // F1 for help
            this.canvas.style.cursor = "help";
        } else {
            // todo
        }
        e.preventDefault();
    }

    private editModeKeyDownHandler(e: KeyboardEvent) {
        if (e.key == "Control" || e.key == "Shift" || e.key == "Alt") {
            this.keyboard.push(e);
            if (e.key == "Control") {
                this.ctrlModeStart();
            } else if (e.key == "Shift") {
                this.shiftModeStart();
            } else {
                this.altModeStart();
            }
        }
    }

    private editModeKeyUpHandler(e: KeyboardEvent) {
        if (e.key == "Control" || e.key == "Shift" || e.key == "Alt") {
            this.keyboard.push(e);
            if (e.key == "Control") {
                this.ctrlModeStop();
            } else if (e.key == "Shift") {
                this.shiftModeStop();
            } else {
                this.altModeStop();
            }
            return;
        }
        if (e.key === "d") {
            if (this.shape.selected_index >= 0) {
                this.shape.delBlob(this.shape.selected_index);
                this.shape.selected_index = -1;
                this.redraw();
            }
            return;
        }

        this.viewModeKeyUpHandler(e);   // Support zoom +/-/= or help ?
        e.preventDefault();
    }

    private registerEventHandlers() {
        this.canvas.addEventListener('mousedown', (e: MouseEvent) => {
            this.mouse.set(e, Canvas.ZOOM_LEVELS[this.zoom_index]);
            if (this.isEditMode())
                this.editModeMouseDownHandler(e);
            else
                this.viewModeMouseDownHandler(e);
        }, false);
        this.canvas.addEventListener('mouseup', (e: MouseEvent) => {
            this.mouse.set(e, Canvas.ZOOM_LEVELS[this.zoom_index]);
            this.canvas.style.cursor = "default";
            if (this.isEditMode())
                this.editModeMouseUpHandler(e);
            else
                this.viewModeMouseUpHandler(e);
            this.mouse.reset(); // Clear left_button_pressed !!!
            e.stopPropagation();
        }, false);
        this.canvas.addEventListener('mouseover', (e: MouseEvent) => {
            this.redraw();
        }, false);
        this.canvas.addEventListener('mousemove', (e: MouseEvent) => {
            this.mouse.set(e, Canvas.ZOOM_LEVELS[this.zoom_index]);
            if (this.isEditMode())
                this.editModeMouseMoveHandler(e);
            else
                this.viewModeMouseMoveHandler(e);
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
            if (this.isEditMode())
                this.editModeKeyDownHandler(e);
            else
                this.viewModeKeyDownHandler(e);
            e.preventDefault();
        }, false);

        this.canvas.addEventListener('keyup', (e: KeyboardEvent) => {
            console.log("Canvas: addEventListener keyup ...", e);
            if (e.key == "Control" || e.key == "Shift" || e.key == "Alt") {
                this.keyboard.push(e);
                if (e.key == "Control") {
                    this.ctrlModeStop();
                } else if (e.key == "Shift") {
                    this.shiftModeStop();
                } else {
                    this.altModeStop();
                }
            } else {
                if (this.isEditMode())
                    this.editModeKeyUpHandler(e);
                else
                    this.viewModeKeyUpHandler(e);
            }
            e.preventDefault();
        }, false);
    }

    setZoom(index: number) {
        // Bad case: (-1 % ZOOM_LEVELS.length) == -1
        if (index < 0)
            index = Canvas.ZOOM_LEVELS.length - 1;
        this.zoom_index = index % Canvas.ZOOM_LEVELS.length;

        this.canvas.width = this.background.naturalWidth * Canvas.ZOOM_LEVELS[this.zoom_index];
        this.canvas.height = this.background.naturalHeight * Canvas.ZOOM_LEVELS[this.zoom_index];

        this.brush.scale(Canvas.ZOOM_LEVELS[this.zoom_index], Canvas.ZOOM_LEVELS[this.zoom_index]);
        this.redraw();

        console.log("Canvas: set zoom: index = ", index, "scale: ", Canvas.ZOOM_LEVELS[index]);
    }

    redraw() {
        this.brush.clearRect(0, 0, this.canvas.width, this.canvas.height);
        if (this.background_loaded) {
            let w = this.background.naturalWidth;
            let h = this.background.naturalHeight;
            this.brush.drawImage(this.background, 0, 0, w, h, 0, 0, w, h);
        }
        this.shape.draw(this.brush);
    }

    fastDrawMovingObject(t: Polygon) {
        let box = t.bbox();
        box.extend(Polygon.LINE_WIDTH * 2);
        this.image_stack.restore(this.brush);
        this.image_stack.save(this.brush, box);

        t.drawDashBorder(this.brush);
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

enum MouseOverStatus {
    DragOver,
    ClickOver,
}

// How to get mouse overStatus out of event handlers ? Record it !
class Mouse {
    start: Point;
    moving: Point;
    stop: Point;
    left_button_pressed: boolean; // mouse moving events erase the overStatus, so define it !

    constructor() {
        this.start = new Point(0, 0);
        this.moving = new Point(0, 0);
        this.stop = new Point(0, 0);

        this.left_button_pressed = false;
    }

    set(e: MouseEvent, scale: number = 1.0) {
        if (e.type == "mousedown") {
            this.start.x = e.offsetX / scale;
            this.start.y = e.offsetY / scale;
            this.left_button_pressed = true;
        } else if (e.type == "mouseup") {
            this.stop.x = e.offsetX / scale;
            this.stop.y = e.offsetY / scale;
        } else if (e.type == "mousemove") {
            this.moving.x = e.offsetX / scale;
            this.moving.y = e.offsetY / scale;
        }
    }

    reset() {
        this.left_button_pressed = false;
    }

    pressed(): boolean {
        return this.left_button_pressed;
    }

    overStatus(): number {
        let d = this.start.distance(this.stop);
        return (this.pressed() && d > Point.THRESHOLD) ?
            MouseOverStatus.DragOver : MouseOverStatus.ClickOver;
    }

    bbox(): Box {
        return Box.bbox(this.start, this.stop);
    }
}

enum KeyboardMode {
    Normal,
    CtrlKeydown,
    CtrlKeyup,
    ShiftKeydown,
    ShiftKeyup,
    AltKeydown,
    AltKeyup
}

// How to get key out of event handlers ?
class Keyboard {
    stack: Array < KeyboardMode > ;

    constructor() {
        this.stack = new Array < KeyboardMode > ();
    }

    push(e: KeyboardEvent) {
        if (e.type == "keydown") {
            if (e.key == "Control")
                this.stack.push(KeyboardMode.CtrlKeydown);
            else if (e.key == "Shift") {
                this.stack.push(KeyboardMode.ShiftKeydown);
            } else if (e.key == "Alt") {
                this.stack.push(KeyboardMode.AltKeydown);
            }
        } else if (e.type == "keyup") {
            if (e.key == "Control")
                this.stack.push(KeyboardMode.CtrlKeyup);
            else if (e.key == "Shift") {
                this.stack.push(KeyboardMode.ShiftKeyup);
            } else if (e.key == "Alt") {
                this.stack.push(KeyboardMode.AltKeyup);
            }
        }
    }

    mode(): KeyboardMode {
        if (this.stack.length < 1)
            return KeyboardMode.Normal;
        return this.stack[this.stack.length - 1];
    }

    pop() {
        if (this.stack.length < 1)
            return;
        let m = this.stack.pop() as KeyboardMode;
        if (m != KeyboardMode.CtrlKeyup && m != KeyboardMode.ShiftKeyup && m != KeyboardMode.AltKeyup) {
            this.stack.push(m); // Restore stack
            return;
        }
        let s = KeyboardMode.Normal; // Search 
        switch (m) {
            case KeyboardMode.CtrlKeyup:
                s = KeyboardMode.CtrlKeydown;
                break;
            case KeyboardMode.ShiftKeyup:
                s = KeyboardMode.ShiftKeydown;
                break;
            case KeyboardMode.AltKeyup:
                s = KeyboardMode.AltKeydown;
                break;
            default:
                break;
        }
        for (let i = this.stack.length - 1; i >= 0; i--) {
            if (s == this.stack[i]) {
                this.stack.splice(i, 1); // pop relative keydown
                break;
            }
        }
    }

    reset() {
        this.stack.length = 0;
    }
}
