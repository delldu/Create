// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

// "use strict";

// Key with timestamp
const module_test = true;

class TimestampKey {
    time: number;
    key: string;

    constructor(key: string) {
        this.time = (new Date()).getTime();
        if (module_test)
            this.time += Math.round(11 * parseInt(key)); // only for test
        this.key = key;
    }
}

class Keyboard {
    duration: number;
    private timer: any;
    private keys: Array < TimestampKey > ;

    constructor(duration: number) {
        this.duration = duration;
        this.timer = setInterval(() => {}, this.duration);
        this.keys = new Array < TimestampKey > ();
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
            if (module_test) {
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
        this.keys.push(new TimestampKey(k));
    }

    clean(): void {
        let now = (new Date()).getTime();
        this.keys = this.keys.filter(c => (now - c.time) < this.duration);
    }
}

if (module_test) {
    console.log("Testing class Keyboard ...")
    let kb = new Keyboard(20); // 20 ms
    for (let k = 0; k < 10; k++) {
        if (module_test)
            kb.push((11 * k).toString());
    }
    kb.start();
}
