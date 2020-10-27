// ***********************************************************************************
// ***
// *** Copyright 2020 Dell(18588220928@163.com), All Rights Reserved.
// ***
// *** File Author: Dell, 2020-09-15 18:09:40
// ***
// ***********************************************************************************

// TsKey -- Key with timestamp
class TsKey {
    time: number;
    key: string;

    constructor(key: string) {
        this.time = (new Date()).getTime();
        this.key = key;
    }
}

class TimeKeyboard {
    duration: number;
    private timer: number;
    private keys: Array < TsKey > ;

    constructor(duration: number) {
        this.duration = duration;
        this.timer = 0;
        this.keys = new Array < TsKey > ();
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
