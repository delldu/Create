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

class ImageFile {

    constructor(id: string) {}

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


// <input type="file" onchange="previewFiles()" multiple>
// <div id="preview"></div>
function previewFiles() {
    var preview = document.querySelector('#preview');
    var files = document.querySelector('input[type=file]').files;

    function readAndPreview(file) {
        if (/\.(jpe?g|png|gif)$/i.test(file.name)) {
            var reader = new FileReader();

            reader.addEventListener("load", function() {
                var image = new Image();
                image.height = 100;
                image.title = file.name;
                image.src = this.result;
                preview.appendChild(image);
            }, false);

            reader.readAsDataURL(file);
        }
    }

    if (files)
        [].forEach.call(files, readAndPreview);
}