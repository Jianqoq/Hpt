import cytoscape from "cytoscape";

export function getTextWidthHeight(text: string, font: string) {
    let canvas = document.createElement("canvas");
    let context = canvas.getContext("2d");
    if (!context) {
        throw new Error("Canvas context not available");
    }
    context.font = font;
    let metrics = context.measureText(text);
    return [metrics.width, metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent];
}

export function createCy(nodes: any, edges: any, container: string) {
    let cy = cytoscape({
        container: document.getElementById(container),
        elements: {
            nodes: nodes,
            edges: edges,
        },
        style: [
            {
                selector: "node[label]",
                style: {
                    label: "data(label)",
                    shape: 'roundrectangle',
                    "border-width": 1,
                    "text-halign": "center",
                    "text-valign": "center",
                },
            },
            {
                selector: "edge",
                style: {
                    "line-color": "#ccc",
                    "target-arrow-color": "#ccc",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                    "font-family": "Arial, sans-serif",
                    "font-size": "14px",
                    "text-wrap": "wrap",
                    "text-opacity": 0,
                },
            },
            {
                selector: 'node[alert]',
                style: {
                    'background-color': 'red',
                    'background-fit': 'cover',
                    'background-width': '100%',
                    'background-height': '100%',
                    'background-opacity': 1
                },
            },
            {
                selector: "node[type = 2][?span_msg]",
                style: {
                    "background-color": "#db4848",
                },
            },
            {
                selector: "edge.clicked",
                style: {
                    "text-opacity": 1,
                },
            },

        ],
        wheelSensitivity: 0.1,
        minZoom: 0,
        maxZoom: 6,
    });

    return cy;
}

function generateRandomId(length: number = 8): string {
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    const charactersLength = characters.length;
    for (let i = 0; i < length; i++) {
        result += characters.charAt(Math.floor(Math.random() * charactersLength));
    }
    return result;
}

export function generateUniqueId(set: Set<string>): string {
    let id = generateRandomId();
    while (set.has(id)) {
        id = generateRandomId();
    }
    return id;
}

export function format_primitive(cy: any) {
    cy.nodes().forEach((node: any) => {
        if (node.data('type') === 1) {
            let data = node.data();
            let shape_property = getTextWidthHeight(data.layout.shape, "14px Arial");
            let strides_property = getTextWidthHeight(data.layout.strides, "14px Arial");
            let dtype_property = getTextWidthHeight(data.dtype, "14px Arial");
            let id_property = getTextWidthHeight(data.id, "14px Arial");
            let max_left_width = Math.max(dtype_property[0], id_property[0]);
            let max_lower_height = Math.max(dtype_property[1], strides_property[1]);
            let max_right_width = Math.max(shape_property[0], strides_property[0]);
            let max_upper_height = Math.max(shape_property[1], id_property[1]);
            let center_width = id_property[0] < 20 ? 20 : id_property[0];
            node.style({
                width: `${max_left_width + max_right_width + center_width + 20 + 10}px`,
                height: `${max_lower_height + max_upper_height + 20}px`
            });
        } else {
            let label = node.data("label");
            let labelproperty = getTextWidthHeight(label, "14px Arial");
            let extraWidth = 20;
            let extraHeight = 20;
            node.style({ "overlay-opacity": 0 });
            node.style("width", labelproperty[0] + extraWidth);
            node.style("height", labelproperty[1] + extraHeight);
        }
    });
    let layout = cy.layout({
        name: 'dagre',
        nodeSep: 100,
        edgeSep: 50,
    });
    layout.run();
}

export function html_label(cy: any, mode: string) {
    if (mode == "raw") {
        return cy.nodeHtmlLabel([
            {
                query: 'node[type = 1]',
                halign: 'center',
                valign: 'center',
                halignBox: 'center',
                valignBox: 'center',
                cssClass: 'html-node-label',
                tpl(data: any) {
                    let shape_property = getTextWidthHeight(data.layout.shape, "14px Arial");
                    let strides_property = getTextWidthHeight(data.layout.strides, "14px Arial");
                    let dtype_property = getTextWidthHeight(data.dtype, "14px Arial");
                    let id_property = getTextWidthHeight(data.id, "14px Arial");
                    let max_left_width = Math.max(dtype_property[0], id_property[0]);
                    let max_lower_height = Math.max(dtype_property[1], strides_property[1]);
                    let max_right_width = Math.max(shape_property[0], strides_property[0]);
                    let max_upper_height = Math.max(shape_property[1], id_property[1]);
                    let center_width = id_property[0] < 20 ? 20 : id_property[0];
                    let content = `
            <div class="html-node-label" style="display: grid; grid-template-areas: 'left1 center right1' 'left1 center right1'; grid-template-columns: ${max_left_width + 10}px ${center_width + 10}px ${max_right_width + 10}px; grid-template-rows: ${max_upper_height + 10}px ${max_lower_height + 10}px;">
                <div
                    style="grid-area: left1; display: flex; justify-content: center; align-items: center; pointer-events: none;">
                    <span>${data.dtype}</span>
                </div>
                <div
                    style="grid-area: right1; display: flex; justify-content: center; align-items: center; pointer-events: none;">
                    <span>${data.layout.shape}</span>
                </div>
                <div class="grid-item border-all" style="grid-area: center; display: flex; justify-content: center; align-items: center; border-left: 1px solid #000000; border-right: 1px solid #000000; pointer-events: none;">
                    <span>${data.id}</span>
                </div>
            </div>
        `;
                    return content; // your html template here
                }
            }
        ]);
    } else if (mode == "mir") {
        return cy.nodeHtmlLabel([
            {
                query: 'node[type = 1]',
                halign: 'center',
                valign: 'center',
                halignBox: 'center',
                valignBox: 'center',
                cssClass: 'html-node-label',
                tpl(data: any) {
                    console.log(data.group_id);
                    let shape_property = getTextWidthHeight(`shape: [${data.layout.shape}]`, "14px Arial");
                    let strides_property = getTextWidthHeight(`strides: [${data.layout.strides}]`, "14px Arial");
                    let dtype_property = getTextWidthHeight(`dtype: [${data.dtype}]`, "14px Arial");
                    let id_property = getTextWidthHeight(data.id, "14px Arial");
                    let fuse_group_property = getTextWidthHeight(`fuse: [${data.fuse_group}]`, "14px Arial");
                    let group = getTextWidthHeight(`group: [${data.group_id}]`, "14px Arial");
                    let max_left_width = Math.max(dtype_property[0], fuse_group_property[0]);
                    let max_lower_height = Math.max(dtype_property[1], strides_property[1]);
                    let max_right_width = Math.max(shape_property[0], strides_property[0]);
                    let max_upper_height = Math.max(shape_property[1], fuse_group_property[1]);
                    let center_width = id_property[0] < 20 ? 20 : id_property[0];
                    let content = `
                        <div class="html-node-label" style="display: grid; grid-template-areas: 'left1 center right1' 'left2 center right2' 'group group group';
                         grid-template-columns: ${max_left_width + 10}px ${center_width + 10}px ${max_right_width + 10}px; grid-template-rows: ${max_upper_height + 10}px ${max_lower_height + 10} ${group[1]}px;">
                            <div
                                style="grid-area: left1; display: flex; justify-content: center; align-items: center; border-bottom: 1px solid #000000; pointer-events: none;">
                                <span>fuse: [${data.fuse_group}]</span>
                            </div>
                            <div
                                style="grid-area: left2; display: flex; justify-content: center; align-items: center; pointer-events: none;">
                                <span>dtype: ${data.dtype}</span>
                            </div>
                            <div
                                style="grid-area: right1; display: flex; justify-content: center; align-items: center; border-bottom: 1px solid #000000; pointer-events: none;">
                                <span>shape: [${data.layout.shape}]</span>
                            </div>
                            <div
                                style="grid-area: right2; display: flex; justify-content: center; align-items: center; pointer-events: none;">
                                <span>strides: [${data.layout.strides}]</span>
                            </div>
                            <div class="grid-item border-all" style="grid-area: center; display: flex; justify-content: center; align-items: center;
                             border-left: 1px solid #000000; border-right: 1px solid #000000; pointer-events: none;">
                                <span>${data.id}</span>
                            </div>
                            <div style="grid-area: group; display: flex; justify-content: center; border-top: 1px solid #000000; align-items: center; pointer-events: none;">
                                <span>group: ${data.group_id}</span>
                            </div>
                        </div>
                    `;
                    return content; // your html template here
                }
            }
        ]);
    }
}

export function format_mir(cy: any) {
    cy.nodes().forEach((node: any) => {
        if (node.data('type') === 1) {
            let data = node.data();
            let shape_property = getTextWidthHeight(`shape: [${data.layout.shape}]`, "14px Arial");
            let strides_property = getTextWidthHeight(`strides: [${data.layout.strides}]`, "14px Arial");
            let dtype_property = getTextWidthHeight(`dtype: [${data.dtype}]`, "14px Arial");
            let id_property = getTextWidthHeight(data.id, "14px Arial");
            let group = getTextWidthHeight(`group: [${data.group_id}]`, "14px Arial");
            let fuse_group_property = getTextWidthHeight(`fuse: [${data.fuse_group}]`, "14px Arial");
            let max_left_width = Math.max(dtype_property[0], fuse_group_property[0]);
            let max_lower_height = Math.max(dtype_property[1], strides_property[1]);
            let max_right_width = Math.max(shape_property[0], strides_property[0]);
            let max_upper_height = Math.max(shape_property[1], fuse_group_property[1]);
            let center_width = id_property[0] < 20 ? 20 : id_property[0];
            node.style({
                width: `${max_left_width + max_right_width + center_width + 30}px`,
                height: `${max_lower_height + max_upper_height + 16 + group[1]}px`
            });
        } else {
            let label = node.data("label");
            let labelproperty = getTextWidthHeight(label, "14px Arial");
            let extraWidth = 20;
            let extraHeight = 20;
            node.style({ "overlay-opacity": 0 });
            node.style("width", labelproperty[0] + extraWidth);
            node.style("height", labelproperty[1] + extraHeight);
        }
    });
    let layout = cy.layout({
        name: 'dagre',
        nodeSep: 200,
        edgeSep: 50,
    });
    layout.run();
}

export function update_layout(cy: any) {
    cy.nodes().forEach((node: any) => {
        let label = node.data("label");
        let labelproperty = getTextWidthHeight(label, "14px Arial");
        let extraWidth = 20;
        let extraHeight = 20;
        node.style({ "overlay-opacity": 0 });
        node.style("width", labelproperty[0] + extraWidth);
        node.style("height", labelproperty[1] + extraHeight);
    });
    let layout = cy.layout({
        name: 'dagre',
        nodeSep: 30,
        edgeSep: 30,
    });
    layout.run();
}