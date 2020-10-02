// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

// <input type="file" id="invisible_file_input" name="files[]" accept="image/*" style="display:none">
// #invisible_file_input { width:0.1px; height:0.1px; opacity:0; overflow:hidden; position:absolute; z-index:-1; }

// if (invisible_file_input) {
//     invisible_file_input.setAttribute('multiple', 'multiple');
//     invisible_file_input.accept = '.jpg,.jpeg,.png,.bmp';
//     invisible_file_input.onchange = project_file_add_local;
//     invisible_file_input.click();
// }

// https://developer.mozilla.org/en-US/docs/Web/API/Event

class NiImage {
    canvas: HTMLCanvasElement;
    image_list: Array < string > ;
    image_index: number;

    constructor(id: string) {
        this.canvas = document.getElementById(id) as HTMLCanvasElement;
        this.canvas.tabIndex = -1; // Support keyboard event

        this.image_list = new Array < string > ();
        this.image_index = -1;

        this.registerEventHandlers();
    }

    registerEventHandlers() {
        // Handle keyboard
        this.canvas.addEventListener('keydown', (e: KeyboardEvent) => {
            e.preventDefault();
        }, false);

        this.canvas.addEventListener('keyup', (e: KeyboardEvent) => {
            if (e.key === "+" || e.key === ">" || e.key === "n" || e.key === "N") {
                this.switchImage(+1);
            } else if (e.key === "-" || e.key === "<" || e.key === "p" || e.key === "P") {
                this.switchImage(-1);
            }
            e.preventDefault();
        }, false);
    }

    addImage(file: File) {
        if (file) {
            let e = document.createElement('img');
            let image_reader = new FileReader();

            image_reader.addEventListener("error", function() {
                //@todo
            }, false);

            image_reader.addEventListener("load", () => {
                e.src = image_reader.result;
                let id = file.name; // spark..., md5sum
                if (this.findImage(id) >= 0)    // duplicate ...
                    return;
                
                e.setAttribute('id', id);
                e.setAttribute('title', '[' + file.name + ']');
                let p = document.getElementById(this.canvas.id);
                if (p) {
                    p.appendChild(e);
                    this.image_list.push(id);
                    this.image_index = this.image_list.length - 1;
                    this.redraw();
                }
            }, false);

            image_reader.readAsDataURL(file);
        }
    }

    deleteImage(id: string) {
        let e = document.getElementById(id);
        if (e && e.parentElement) {
            e.parentElement.removeChild(e);
            this.switchImage(0);
            this.redraw();
        }
    }

    switchImage(delta: number) {
        let n = this.image_list.length;
        if (n > 0 && Math.abs(delta) <= 1) {
            this.image_index = (n + this.image_index + delta) % n;
            this.redraw();
        }
    }

    findImage(id: string): number {
        return this.image_list.indexOf(id);
    }

    redraw() {
        let brush = this.canvas.getContext('2d') as CanvasRenderingContext2D;
        if (!brush)
            return;

        brush.clearRect(0, 0, this.canvas.width, this.canvas.height);
        // Draw image ...

        // console.log("image -- ", this.image_index, this.image_list);
        if (this.image_index >= 0 && this.image_index < this.image_list.length) {
            let e = document.getElementById(this.image_list[this.image_index]);
            if (!e) {
                // console.log("e === null, this is error !!!");
                return;
            }
            let background = (e as HTMLImageElement);
            if (background) {
                brush.drawImage(background, 0, 0);
            } else {
                console.log("background === null, this is error !!!");
            }
        } else {
            console.log("What's wrong ? image -- ", this.image_index, this.image_list);
        }
    }

    open(id: string) {
        // https://developer.mozilla.org/en-US/docs/Web/API/FileList
        const element = document.getElementById(id) as HTMLInputElement;
        element.setAttribute('multiple', 'multiple');
        element.accept = '.jpg,.jpeg,.png,.bmp';
        element.addEventListener('change', (event: Event) => {
            let files = (event.target as HTMLInputElement).files;
            if (!files)
                return;
            for (let i = 0; i < files.length; i++) {
                let image_reader = new FileReader();
                image_reader.addEventListener("error", (e: Event) => {
                    //@todo
                }, false);
                image_reader.addEventListener("load", (e: Event) => {
                    // x.src = image_reader.result;
                }, false);
                image_reader.readAsDataURL(files[i]);
            }
        }, false);
        // element.click();
    }

    clean(src: ImageData): ImageData {
        let dst = new ImageData(src.data, src.width, src.height);

        return dst;
    }

    color(src: ImageData): ImageData {
        let dst = new ImageData(src.width, src.height);
        return dst;
    }

    zoom(src: ImageData, scale: number): ImageData {
        let dst = new ImageData(src.width, src.height);
        return dst;
    }

    patch(src: ImageData, mask: ImageData): ImageData {
        let dst = new ImageData(src.width, src.height);
        return dst;
    }

    save(image: ImageData, filename: string) {

    }

    upload(image: ImageData, url: string) {

    }
}