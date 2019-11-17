class Palette {
    constructor() {
        this.colors = {
            navy: '#001F3F',
            blue: '#0074D9',
            aqua: '#7FDBFF',
            teal: '#39CCCC',
            olive: '#3D9970',
            green: '#2ECC40',
            lime: '#01FF70',
            yellow: '#FFDC00',
            orange: '#FF851B',
            red: '#FF4136',
            maroon: '#85144b',
            fuchsia: '#F012BE',
            purple: '#B10DC9',
            black: '#111111',
            gray: '#AAAAAA',
            silver: '#DDDDDD',
            white: '#FFFFFF',
        };
        this.available = Object.keys(this.colors);
    }
    getColor(col) {
        return color(this.colors[col]);
    }
}
