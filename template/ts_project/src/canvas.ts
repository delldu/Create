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

    // clone(): Point {
    //     return new Point(this.x, this.y);
    // }

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

    valid(): boolean {
        // console.log("Box: too small box, give up.");
        return (this.w >= Point.THRESHOLD * 4 && this.h >= Point.THRESHOLD * 4);
    }

    size(): number {
        return this.w * this.w;
    }

    // clone(): Box {
    //     return new Box(this.x, this.y, this.w, this.h);
    // }
    //
    // offset(deltaX: number, deltaY: number) {
    //     this.x += deltaX;
    //     this.y += deltaY;
    // }
    //
    // zoom(s: number) {
    //     this.x *= s;
    //     this.y *= s;
    //     this.w *= s;
    //     this.h *= s;
    // }

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
        if (box.x > this.x + this.w)
            return false;
        if (this.x > box.x + box.w)
            return false;
        if (box.y > this.y + this.h)
            return false;
        if (this.y > box.y + box.h)
            return false;
        return true;
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

    // Set 2x2 polygon via rectangle
    set2x2(box: Box) {
        this.reset();
        this.push(new Point(box.x, box.y));
        this.push(new Point(box.x, box.y + box.h));
        this.push(new Point(box.x + box.w, box.y + box.h));
        this.push(new Point(box.x + box.w, box.y));
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
            if (p.distance(this.points[i]) <= Point.THRESHOLD)
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

    // s is a JSON string
    load(s: string) {
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
                    for (let k3 in j[k1][k2]) {
                        if (!j[k1][k2].hasOwnProperty(k3))
                            continue;
                        // k3 == "label" || "points" ...
                        if (k3 == "label") {
                            blob.label = parseInt(j[k1][k2][k3]);
                            continue;
                        }
                        if (k3 != "points")
                            continue;
                        // now k3 === points
                        for (let k4 in j[k1][k2][k3]) {
                            if (!j[k1][k2][k3].hasOwnProperty(k4))
                                continue;
                            blob.push(new Point(j[k1][k2][k3][k4].x, j[k1][k2][k3][k4].y));
                        }
                        this.push(blob);
                    }
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

    // Region draw speed up drawing ...
    regionDraw(brush: CanvasRenderingContext2D, rect: Box) {
        for (let i = 0; i < this.blobs.length; i++) {
            let bbox = this.blobs[i].bbox();
            if (!rect.intersect(bbox))
                continue;
            this.blobs[i].draw(brush, (i == this.selected_index));
        }
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

    // if return true -- need redraw, else skip
    clicked(ctrl: boolean, m: Mouse): boolean {
        // vertex be clicked ?
        let [v_index, v_sub_index] = this.findVertex(m.start);
        if (v_index >= 0 && v_sub_index >= 0) {
            if (ctrl) {
                // Delete this vertex !!!
                this.blobs[v_index].delPoint(v_sub_index);
                // Delete more ... whole blob ?
                if (this.blobs[v_index].points.length < 3)
                    this.delBlob(v_index);
                return true;
            } else {
                console.log("Shape: want to delete vertex ? press ctrl key and click vertex.")
            }
            return false;
        }

        // edge be clicked with ctrl ?
        let [e_index, e_sub_index] = this.findEdge(m.start);
        if (e_index >= 0 && e_sub_index >= 0) {
            // Add vertex
            if (ctrl) {
                this.blobs[e_index].addPoint(e_sub_index + 1, m.start);
                return true;
            } else {
                console.log("Shape: want to add vertex ? press ctrl key and click edge.")
            }
            return false;
        }

        // whole blob be clicked ?
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
    dragging(ctrl: boolean, m: Mouse, brush: CanvasRenderingContext2D, image_stack: ImageStack) {
        // vertex could be dragged ?

        // drag vertex ?
        let [v_index, v_sub_index] = this.findVertex(m.start);
        if (v_index >= 0 && v_sub_index >= 0) {
            let n = this.blobs[v_index].points.length;
            let blob = new Polygon();
            // Bad case -1 % n = -1 !!!
            // So we use (n + v_sub_index - 1)%n instead of (v_sub_index - 1) % n
            blob.push(this.blobs[v_index].points[(n + v_sub_index - 1) % n]);
            blob.push(this.blobs[v_index].points[(v_sub_index + 1) % n]);
            blob.push(m.moving);

            let rect = blob.bbox();
            if (!rect.valid())
                return;

            image_stack.restore(brush);
            rect.extend(Polygon.LINE_WIDTH);
            image_stack.save(brush, rect);

            blob.drawDashBorder(brush);

            return;
        }

        // drag whole blob ?
        let b_index = this.findBlob(m.start);
        if (b_index >= 0) {
            let deltaX = m.moving.x - m.start.x;
            let deltaY = m.moving.y - m.start.y;
            let blob = this.blobs[b_index].clone();
            blob.offset(deltaX, deltaY);

            let rect = blob.bbox();
            if (!rect.valid())
                return;

            image_stack.restore(brush);
            rect.extend(Polygon.LINE_WIDTH);
            image_stack.save(brush, rect);

            blob.drawDashBorder(brush);
            return;
        } else {
            // Add new blob
            let box = m.moving_bbox();
            if (!box.valid())
                return;

            let blob = new Polygon();
            if (ctrl) { // Add 3x3 rectangle, heavy blob
                blob.set3x3(box);
            } else { // Add 2x2 rectangle, light blob
                blob.set2x2(box);
            }

            let rect = blob.bbox();
            if (!rect.valid())
                return;

            image_stack.restore(brush);
            rect.extend(Polygon.LINE_WIDTH);
            image_stack.save(brush, rect);

            blob.drawDashBorder(brush);
            return;
        } // end of b_index
    }

    // return true -- need redraw, else false
    dragged(ctrl: boolean, m: Mouse): boolean {
        // vertex could be dragged ? this will change blob
        let [v_index, v_sub_index] = this.findVertex(m.start);
        if (v_index >= 0 && v_sub_index >= 0) {
            if (ctrl) {
                let deltaX = m.stop.x - m.start.x;
                let deltaY = m.stop.y - m.start.y;
                this.blobs[v_index].points[v_sub_index].x += deltaX;
                this.blobs[v_index].points[v_sub_index].y += deltaY;
                return true;
            } else {
                console.log("Shape: want dragging vertex ? press ctrl key and drag vertex.")
            }
            return false;
        }

        // whole blob could be dragged ?
        let b_index = this.findBlob(m.start);
        if (b_index >= 0) {
            let deltaX = m.stop.x - m.start.x;
            let deltaY = m.stop.y - m.start.y;
            this.blobs[b_index].offset(deltaX, deltaY);
        } else {
            // Add new blob
            let box = m.bbox();
            if (!box.valid())
                return false;

            let blob = new Polygon();
            if (ctrl) { // Add 3x3 rectangle, heavy blob
                blob.set3x3(box);
            } else { // Add 2x2 rectangle, light blob
                blob.set2x2(box);
            }
            box = blob.bbox();
            if (!box.valid())
                return false;

            this.push(blob);
        } // end of b_index
        return true;
    } // end of dragged
}

class Canvas {
    static ZOOM_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4, 5, 6, 7, 8, 9, 10];
    static DEFAULT_ZOOM_LEVEL = 3; // 1.0 index

    canvas: HTMLCanvasElement; // canvas element
    private brush: CanvasRenderingContext2D;

    private background: HTMLImageElement; // Image;
    private background_loaded: boolean;
    shape_blobs: Shape;
    shift_blob: Polygon;

    mode_index: number;

    // Polygon container
    private image_stack: ImageStack;

    // Zoom control
    zoom_index: number;

    // Handle mouse
    private mouse: Mouse;
    private keyboard: Keyboard;

    // Key
    key: string;

    constructor(id: string) {
        this.canvas = document.getElementById(id) as HTMLCanvasElement;
        this.canvas.tabIndex = -1; // Support keyboard event

        this.brush = this.canvas.getContext('2d') as CanvasRenderingContext2D;

        this.shape_blobs = new Shape();
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

        // Create background for canvas
        // this.background = document.createElement('img') as HTMLImageElement;
        // this.background.id = this.canvas.id + "_background_image";
        // this.background.style.display = "none";
        // this.canvas.appendChild(this.background);
        this.background = new Image() as HTMLImageElement;
        // this.background.src = "dog.jpg";
        this.background_loaded = false;

        this.image_stack = new ImageStack();

        this.key = "";
    }

    reset() {
        this.shape_blobs.reset();
        this.shift_blob.reset();
        this.zoom_index = Canvas.DEFAULT_ZOOM_LEVEL;
        this.mode_index = 0;
        this.image_stack = new ImageStack();
        this.background_loaded = false;
        this.key = "";
    }

    // loadBackgroundFromFile(file: File) {
    //     this.background_loaded = false;
    //     let reader = new FileReader();
    //     reader.addEventListener("load", () => {
    //         this.background.src = reader.result;
    //         this.background_loaded = true;
    //         if (this.canvas.width < this.background.naturalWidth)
    //             this.canvas.width = this.background.naturalWidth;
    //         if (this.canvas.height < this.background.naturalHeight)
    //             this.canvas.height = this.background.naturalHeight;
    //
    //         this.setZoom(Canvas.DEFAULT_ZOOM_LEVEL);
    //     }, false);
    //     reader.readAsDataURL(file);
    // }

    // loadBackground(e: HTMLImageElement) {
    //     this.background_loaded = false;
    //     this.background = e;
    //     this.canvas.width = this.background.naturalWidth;
    //     this.canvas.height = this.background.naturalHeight;
    //     this.setZoom(Canvas.DEFAULT_ZOOM_LEVEL);
    //     this.background_loaded = true;
    // }

    loadBackground(dataurl: string) {
        this.background_loaded = false;
        this.background.src = dataurl;
        this.canvas.width = this.background.naturalWidth;
        this.canvas.height = this.background.naturalHeight;
        this.setZoom(Canvas.DEFAULT_ZOOM_LEVEL);
        this.background_loaded = true;
    }

    loadBlobs(blobs: string) {
        this.shape_blobs.load(blobs);
    }

    saveBlobs(): string {
        return JSON.stringify(this.shape_blobs, undefined, 2);
    }

    setMode(index: number) {
        // Bad case: (-1 % 2) == -1
        if (index < 0)
            index = 1;
        index = index % 2;
        this.mode_index = index;
        console.log("Canvas: set mode:", this.mode_index);
    }

    getMode(): number {
        return this.mode_index;
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
    ctrlModeStartHandle() {
        // nothing to do
    }

    ctrlModeMouseMoveHandle() {

    }

    ctrlModeMouseUpHandle() {
        if (this.mouse.overStatus() == MouseOverStatus.ClickOver) {
            // vertex be clicked ?
            let [v_index, v_sub_index] = this.shape_blobs.findVertex(this.mouse.start);
            if (v_index >= 0 && v_sub_index >= 0) {
                // Delete this vertex !!!
                this.shape_blobs.blobs[v_index].delPoint(v_sub_index);
                // Delete more ... whole blob ?
                if (this.shape_blobs.blobs[v_index].points.length < 3)
                    this.shape_blobs.delBlob(v_index);
                this.redraw();
                return;
            }

            // edge be clicked with ctrl ?
            let [e_index, e_sub_index] = this.shape_blobs.findEdge(this.mouse.start);
            if (e_index >= 0 && e_sub_index >= 0) {
                // Add vertex
                this.shape_blobs.blobs[e_index].addPoint(e_sub_index + 1, this.mouse.start);
                this.redraw();
                return;
            }
            // other case, ignore
        } else if (this.mouse.overStatus() == MouseOverStatus.DragOver) {

        }
    }

    ctrlModeStopHandle() {
        this.keyboard.pop();
    }

    shiftModeStartHandle() {
        this.shift_blob.reset();
    }

    shiftModeMouseMoveHandle() {

    }

    shiftModeMouseUpHandle() {
        // Click, add one point;, drag, add two points
        if (this.mouse.overStatus() == MouseOverStatus.ClickOver) {
            this.shift_blob.push(this.mouse.stop);
        } else if (this.mouse.overStatus() == MouseOverStatus.DragOver) {
            this.shift_blob.push(this.mouse.start);
            this.shift_blob.push(this.mouse.stop);
        }
        // other case, simple ignore
    }

    shiftModeStopHandle() {
        // is valid ?
        if (this.shift_blob.points.length >= 3) {
            this.shape_blobs.push(this.shift_blob);
            this.redraw();
        }
        this.keyboard.pop();
    }

    altModeStartHandle() {
        // todo
    }

    altModeMouseMoveHandle() {
        // todo
    }

    altModeMouseUpHandle() {
        // todo
    }

    altModeStopHandle() {
        // todo
        this.keyboard.pop();
    }

    normalModeMouseMoveHandle() {
        if (this.mouse.pressed()) {
            // this.shape_blobs.dragging(e.ctrlKey, this.mouse, this.brush, this.image_stack);
        }
    }

    normalModeMouseUpHandle() {
        if (this.mouse.overStatus() == MouseOverStatus.ClickOver) {
            let b_index = this.shape_blobs.findBlob(this.mouse.start);
            if (b_index >= 0) {
                // Selected/remove selected flag
                if (this.shape_blobs.selected_index == b_index) {
                    this.shape_blobs.selected_index = -1;
                } else {
                    this.shape_blobs.selected_index = b_index;
                }
                this.redraw();
            }
            // if (this.shape_blobs.clicked(e.ctrlKey, this.mouse))
            //     this.redraw();
        } else if (this.mouse.overStatus() == MouseOverStatus.DragOver) {
            // whole blob could be dragged ?
            let b_index = this.shape_blobs.findBlob(this.mouse.start);
            if (b_index >= 0) {
                let deltaX = this.mouse.stop.x - this.mouse.start.x;
                let deltaY = this.mouse.stop.y - this.mouse.start.y;
                this.shape_blobs.blobs[b_index].offset(deltaX, deltaY);
            } else {
                // Add new blob
                let box = this.mouse.bbox();
                if (!box.valid())
                    return;

                let blob = new Polygon();
                blob.set2x2(box);
                this.shape_blobs.push(blob);
                this.redraw();
            } // end of b_index

            // if (this.shape_blobs.dragged(e.ctrlKey, this.mouse))
            //     this.redraw();
        }
    }

    private viewModeMouseDownHandler(e: MouseEvent) {
        // console.log("viewModeMouseDownHandler ...", e);
    }

    private viewModeMouseMoveHandler(e: MouseEvent) {
        if (this.mouse.pressed()) {
            this.canvas.style.cursor = "pointer";
            // this.canvas.start = new Point(this.mouse.stop.x, this.mouse.stop.y);
        } else {
            this.canvas.style.cursor = "default";
        }
    }

    private viewModeMouseUpHandler(e: MouseEvent) {
        if (this.mouse.overStatus() == MouseOverStatus.DragOver) {
            console.log("Canvas: dragging ..., which src object ? moving background ?");
        }
    }

    private editModeMouseDownHandler(e: MouseEvent) {
        // Start:  On selected vertex, On selected Edge, Inside, Blank area
        // e.ctrlKey, e.altKey -- boolean

        this.image_stack.reset();
    }

    private editModeMouseMoveHandler(e: MouseEvent) {
        // Set mouse pointer
        this.canvas.style.cursor = this.mouse.pressed() ? "pointer" : "default";

        if (this.keyboard.mode() == KeyboardMode.CtrlKeydown) {
            this.ctrlModeMouseMoveHandle();
        } else if (this.keyboard.mode() == KeyboardMode.ShiftKeydown) {
            this.shiftModeMouseMoveHandle();
        } else if (this.keyboard.mode() == KeyboardMode.AltKeydown) {
            this.altModeMouseMoveHandle();
        } else {
            // Normal mode
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
        // console.log("viewModeKeydownHandler ...");
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
            // todo
        } else {
            // todo
        }
        e.preventDefault();
    }

    private editModeKeyDownHandler(e: KeyboardEvent) {
        if (e.key == "Control" || e.key == "Shift" || e.key == "Alt") {
            this.keyboard.push(e);
            if (e.key == "Control") {
                this.ctrlModeStartHandle();
            } else if (e.key == "Shift") {
                this.shiftModeStartHandle();
            } else {
                this.altModeStartHandle();
            }
        }
    }

    private editModeKeyUpHandler(e: KeyboardEvent) {
        if (e.key == "Control" || e.key == "Shift" || e.key == "Alt") {
            this.keyboard.push(e);
            if (e.key == "Control") {
                this.ctrlModeStopHandle();
            } else if (e.key == "Shift") {
                this.shiftModeStopHandle();
            } else {
                this.altModeStopHandle();
            }
            return;
        }
        if (e.key === "d") {
            if (this.shape_blobs.selected_index >= 0) {
                console.log("Canvas: delete blob:", this.shape_blobs.selected_index);
                this.shape_blobs.delBlob(this.shape_blobs.selected_index);
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
            this.mouse.set(e, Canvas.ZOOM_LEVELS[this.zoom_index]);
            if (this.isEditMode())
                this.editModeMouseDownHandler(e);
            else
                this.viewModeMouseDownHandler(e);
        }, false);
        this.canvas.addEventListener('mouseup', (e: MouseEvent) => {
            // Every this is calm ...
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
                    this.ctrlModeStopHandle();
                } else if (e.key == "Shift") {
                    this.shiftModeStopHandle();
                } else {
                    this.altModeStopHandle();
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
        if (index < 0) {
            index = Canvas.ZOOM_LEVELS.length - 1;
        }
        index = index % Canvas.ZOOM_LEVELS.length;

        this.zoom_index = index;

        this.canvas.width = this.background.naturalWidth * Canvas.ZOOM_LEVELS[this.zoom_index];
        this.canvas.height = this.background.naturalHeight * Canvas.ZOOM_LEVELS[this.zoom_index];

        this.brush.scale(Canvas.ZOOM_LEVELS[this.zoom_index], Canvas.ZOOM_LEVELS[this.zoom_index]);
        this.redraw();

        console.log("Canvas: set zoom: index = ", index, "scale: ", Canvas.ZOOM_LEVELS[index]);
    }

    redraw() {
        this.brush.clearRect(0, 0, this.canvas.width, this.canvas.height);
        // Draw image ...
        if (this.background_loaded) {
            let w = this.background.naturalWidth;
            let h = this.background.naturalHeight;

            this.brush.drawImage(this.background, 0, 0, w, h, 0, 0, w, h);
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