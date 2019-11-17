let num_nodes = 400; // How many nodes
let num_edges = 400;
let activity_log = [];
let graph;
let nodes = [];
let links = [];
let spread_prob; // Probability of spreading infection
let buffer = 100; // Leave some space around the network
let ID = 0; // Used to map nodes to rows of the adjacency matrix
let adj; // Adjacency matrix
let node_diameter = 10; // How big should the nodes be?
let focused = true;
let locality_range = 100;
let min_locality = locality_range / 2;
let max_locality = locality_range * 1.2;
let cur_locality = min_locality;
let window_size = 1000;
let attraction = 1;
let repulsion = 1;
// Palette
let spikeColor;
let restColor;

// Interface
let slider;
let max_signals = 10000; // Limit number of infected agents to avoid killing my computer
let cur_infected = 0; // How many infected agents are travelling
let last_additions = [];

function mousePressed() {
    for (let node of nodes) {
        let d = dist(mouseX, mouseY, node.pos.x, node.pos.y);
        if (d < 10) {
            node.infect(true);
        }
    }
}

function makeNodes() {
    nodes = [];
    ID = 0;

    for (let i = 0; i < num_nodes; i++) {
        let x = buffer / 2 + random(width - buffer);
        let y = buffer / 2 + random(height - buffer * 2);
        let pos = createVector(x, y);
        nodes.push(new Node(pos, ID));
        ID += 1;
    }
}

function localWire() {
    locality = slider.value();
    links = [];

    for (let node of nodes) {
        node.out = [];
        node.in = [];
    }

    let link_pairs = [];
    for (let from of nodes) {
        for (let to of nodes) {
            if (from.id < to.id) {
                let d = dist(from.pos.x, from.pos.y, to.pos.x, to.pos.y);
                if (d < locality) {
                    let link = new Link(from, to);
                    link_pairs.push(link); // to iterate over them
                    link = new Link(to, from);
                    link_pairs.push(link); // to iterate over them
                }
            }
        }
    }
    links = [];
    for (let i = 0; i < link_pairs.length; i += 2) {
        let link;
        if (random() < 0.5) {
            link = link_pairs[i];
        } else {
            link = link_pairs[i + 1];
        }
        link.from.out.push(link);
        link.to.in.push(link);
        links.push(link);
    }
}

// function probWire() {
//     locality = slider.value();

//     let from = random(nodes);
//     let candidates = [];

//     for (let to of nodes) {
//         if (from.id != to.id) {
//             let d = dist(from.pos.x, from.pos.y, to.pos.x, to.pos.y);
//             if (d < locality) {
//                 candidates.push(to);
//             }
//         }
//     }
//     let to = random(candidates);
//     let link = new Link(from, to);
//     link.from.out.push(link);
//     link.to.in.push(link);
//     links.push(link);

//     last_additions.push(new Link(from, to));
// }
function probWire() {
    locality = slider.value();
    links = [];

    adj = Array(num_nodes)
        .fill()
        .map(() => Array(num_nodes).fill(0));

    for (let node of nodes) {
        node.out = [];
        node.in = [];
    }
    for (let i = 0; i < num_edges; i++) {
        let from = random(nodes);
        let candidates = [];

        for (let to of nodes) {
            if (from.id != to.id) {
                let d = dist(from.pos.x, from.pos.y, to.pos.x, to.pos.y);
                if (d < locality) {
                    candidates.push(to);
                }
            }
        }
        if (candidates.length > 0) {
            let to = random(candidates);
            if (adj[from.id][to.id] == 0) {
                let link = new Link(from, to);
                link.from.out.push(link);
                link.to.in.push(link);
                links.push(link);
                adj[from.id][to.id] = 1;
            }
        }
    }
}

function rewire() {
    let to_cut = [];
    let to_add = [];
    let to_keep = [];

    for (let edge of links) {
        let len = p5.Vector.dist(edge.from.pos, edge.to.pos);
        if (len > locality) {
            to_cut.push(edge);
        } else {
            to_keep.push(edge);
        }
    }
}

function setup() {
    spikeColor = color(255, 0, 50);
    restColor = color(0, 65, 225);
    let cnv = createCanvas(1200, 800);
    let reset = createButton('reset');
    reset.mousePressed(() => {
        cur_infected = 0;
        makeNodes();
        localWire();
        plotNetwork();
    });
    reset.position(400, 5);
    reset.style('width', '80px');
    let spark = createButton('spark');
    spark.mousePressed(() => {
        for (let node of nodes) {
            node.infect(true);
        }
    });
    spark.position(500, 5);
    spark.style('width', '80px');
    graph = createGraphics(width, height);

    slider = createSlider(min_locality, max_locality, cur_locality, 1); // range[0-100] start from 55, increments of 5
    slider.input(() => {
        cur_infected = 0;
        localWire();
        plotNetwork();
    });
    slider.position(100, 5);
    slider.style('width', '100px');

    makeNodes();
    localWire();
    plotNetwork();
}
function plotNetwork() {
    graph.clear();
    for (let link of links) {
        link.show(graph);
    }
    for (let node of nodes) {
        node.show(graph);
    }
}
function draw() {
    background(50);
    image(graph, 0, 0);

    // Draw FPS (rounded to 2 decimal places) at the bottom left of the screen
    let fps = frameRate();
    fill(255);
    stroke(0);
    strokeWeight(1);
    textSize(15);
    text('FPS: ' + fps.toFixed(0), width - 100, 20);

    cur_locality = int(slider.value());
    stroke(0);
    strokeWeight(2);
    fill(220, 220, 220);
    textSize(20);
    text('r : ' + cur_locality, 20, 20);
    text(
        'M : ' + num_edges,
        width - max_locality - 250,
        height - max_locality + 20,
    );
    text('Signals : ' + cur_infected, 210, 20);

    for (let node of nodes) {
        node.animate();
    }

    for (let node of nodes) {
        node.update();
    }

    for (let link of links) {
        link.animate();
        link.update();
    }
    activity_log.unshift(cur_infected);
    if (activity_log.length > width) {
        activity_log.pop();
    }
    noFill();
    // indicator is between 0 and 1
    let indicator = cur_infected / max_signals;
    stroke(255 * indicator, (1 - indicator) * 255, 0);
    beginShape();
    for (let idx in activity_log) {
        const level = activity_log[idx];
        vertex(idx, height - (level / max_signals) * buffer * 1.5);
    }
    endShape();
    stroke(255, 50);
    line(0, height - 1, width, height - 1);
    if (cur_infected <= 0) {
        slider.value(slider.value() + 5);
        localWire();
        // probWire();
        plotNetwork();
        for (let node of nodes) {
            node.infect(true);
        }
    }
    if (cur_locality == max_locality) {
        num_edges *= 1.1;
        num_edges = int(num_edges);
        slider.value(min_locality);
    }
    // for (let last_addition of last_additions) {
    //     stroke('yellow');
    //     strokeWeight(10);
    //     line(
    //         last_addition.from.pos.x,
    //         last_addition.from.pos.y,
    //         last_addition.to.pos.x,
    //         last_addition.to.pos.y,
    //     );
    // }
    stroke(255, 255, 0);
    strokeWeight(2);
    ellipse(width - max_locality, height - max_locality, cur_locality * 2);
}

class Node {
    constructor(pos, ID) {
        this.pos = pos;
        this.id = ID;
        ID += 1;
        this.out = [];
        this.in = [];
        this.has_been_infected = false;
        this.threshold = 2;
        this.base_threshold = 4;
        this.activity = 0;
        this.spike_time = 0;
    }
    // force is used so that clicking bypasses the infection probability
    infect(force) {
        this.activity += 1;
        if (force || this.activity > this.threshold) {
            this.spike_time = new Date().getTime();
            this.activity = -this.threshold;
            if (!force) {
                this.threshold += 1;
            }
            for (let edge of this.out) {
                // if we have not reached the max amount of carriers add a new one
                if (cur_infected < max_signals) {
                    edge.infect();
                    cur_infected += 1;
                }
            }

            // for (let edge of this.in) {
            //     if (abs(this.spike_time - edge.from.spike_time) < window_size) {
            //         if (
            //             p5.Vector.dist(this.pos, edge.from.pos) >
            //             node_diameter * 2
            //         ) {
            //             let dir = p5.Vector.sub(edge.from.pos, this.pos);
            //             dir.setMag(attraction);
            //             this.pos.add(dir);
            //         }
            //     } else {
            //         let dir = p5.Vector.sub(this.pos, edge.from.pos);
            //         dir.setMag(repulsion);
            //         this.pos.add(dir);
            //     }
            // }
        }
    }

    update() {
        if (this.threshold > this.base_threshold) {
            this.threshold -= 0.1;
        }
        if (this.activity > 0) {
            this.activity -= 0.005;
        }
    }

    show(p) {
        // p.noStroke(); // no border
        // p.fill(0, 150, 10);
        // p.ellipse(this.pos.x, this.pos.y, node_diameter, node_diameter);
    }
    animate() {
        noStroke();
        if (this.activity > 0) {
            let col = lerpColor(
                restColor,
                spikeColor,
                this.activity / this.threshold,
            );
            fill(col);
            ellipse(this.pos.x, this.pos.y, node_diameter + this.activity * 5);
        } else {
            fill(restColor);
            ellipse(this.pos.x, this.pos.y, node_diameter);
        }
        noFill();
        stroke(100);
        strokeWeight(1);
        ellipse(this.pos.x, this.pos.y, node_diameter + this.threshold * 5);
    }
}

// to show the direction of the link, require some calculation but it's worth it.
function arrowhead(from, to, base, height, distance, p) {
    let rot = p5.Vector.sub(to.pos, from.pos).heading(); // get angle of the link
    p.push();
    p.translate(to.pos.x, to.pos.y); // move the origin to the target the arrow is pointing to
    p.rotate(rot); // rotate to align the tip
    p.noStroke(); // strong independent arrows need no border
    p.fill(255, 50); // a bit of transparency, we want to see them if they overlap
    p.triangle(
        -distance / 2 - 1,
        0,
        -distance / 2 - height,
        -base,
        -distance / 2 - height,
        +base,
    );
    p.pop();
}

class Link {
    constructor(from, to) {
        this.from = from;
        this.to = to;
        this.carriers = [];
        this.dt = 3; // step length along the link, longer = faster
        this.length = p5.Vector.dist(this.from.pos, this.to.pos);
        this.activity = 0;
    }

    infect() {
        this.activity += 1;
        this.carriers.push(0);
    }

    show(p) {
        p.strokeWeight(1);
        // paint link red if there are carriers travelling on it
        if (this.carriers.length == 0) {
            p.stroke(255, 50); // default is greyish
        } else {
            p.stroke(255, 50, 50); // infected is reddish
        }

        p.line(this.from.pos.x, this.from.pos.y, this.to.pos.x, this.to.pos.y);
        arrowhead(this.from, this.to, 2, 6, node_diameter, p);
    }

    animate() {
        for (let carrier of this.carriers) {
            let tmp = p5.Vector.lerp(
                this.from.pos,
                this.to.pos,
                carrier / this.length,
            );
            noStroke();
            fill(255);
            ellipse(tmp.x, tmp.y, 5, 5);
            // noFill();
            // stroke(0, 30);
            // ellipse(this.from.pos.x, this.from.pos.y, carrier * 2, carrier * 2);
        }
    }

    update() {
        let new_carriers = [];

        for (let carrier of this.carriers) {
            // move
            let new_traveler = carrier + this.dt;
            if (new_traveler < this.length) {
                new_carriers.push(new_traveler);
            } else {
                // destination reached
                this.to.infect(false);
                cur_infected -= 1;
            }
        }
        this.carriers = new_carriers;
    }
}
