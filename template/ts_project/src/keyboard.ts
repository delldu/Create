// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

// TsKey -- Key with timestamp

const keyboard_test = true;
class TsKey {
    time: number;
    key: string;

    constructor(key: string) {
        this.time = (new Date()).getTime();
        if (keyboard_test)
            this.time += Math.round(11 * parseInt(key)); // only for test
        this.key = key;
    }
}

class Keyboard {
    duration: number;
    private timer: number;
    private keys: Array < TsKey > ;

    constructor(duration: number) {
        this.duration = duration;
        this.timer = 0;
        this.keys = new Array < TsKey > ();

        if (keyboard_test) {
            console.log("This is test version, please set keyboard_test = false for production .");
        }
    }

    reset() {
        this.keys.length = 0;
    }

    getKeys(): Array < string > {
        let list = [];
        for (let c of this.keys)
            list.push(c.key);
        return list;
    }

    start() {
        this.stop();
        this.timer = setInterval(() => {
            this.clean();
            if (keyboard_test) {
                console.log(this.getKeys());
                if (this.keys.length < 1)
                    this.stop();
            }
        }, this.duration);
    }

    stop() {
        if (this.timer)
            clearInterval(this.timer);
    }

    push(k: string) {
        this.keys.push(new TsKey(k));
    }

    clean(): void {
        this.stop();
        let now = (new Date()).getTime();
        this.keys = this.keys.filter(c => (now - c.time) < this.duration);
        this.start();
    }
}

if (keyboard_test) {
    console.log("Testing class Keyboard ...")
    let kb = new Keyboard(20); // 20 ms
    for (let k = 0; k < 10; k++) {
        if (keyboard_test)
            kb.push((11 * k).toString());
    }
    kb.start();
}
