import cytoscape from "cytoscape";
import { generateUniqueId } from "../GraphUtils";
import { to_op_str } from "../op";

export enum NodeType {
    Block,
    Primitive,
    Op,
}

export enum BlockType {
    Function,
    If,
    While,
    WhileCond,
    WhileBody,
    For,
    Unknown,
}

export interface BlockNode {
    id: string;
    label: string;
    block_id: number;
    type: NodeType;
    block_type: BlockType;
    preds?: Set<string>;
    parent?: string;
    expanded: boolean;
    inputs: Set<string>;
    outputs: Set<string>;
}

export interface PrimitiveNode {
    id: string;
    label: string;
    inputs: Array<string>;
    block_id: number;
    layout: { shape: Array<number>, strides: Array<number> };
    type: NodeType;
    dtype: string;
    parent_block?: any;
    fuse_group?: Array<number>;
    group_id?: number;
    is_const?: boolean;
}

export interface OpNode {
    id: string;
    label: string;
    inputs: Array<string>;
    result_id: string;
    block_id: number;
    type: NodeType;
    parent_block?: any;
    span_msg: string | null;
}

export interface RustNode {
    block_id: number;
    const_val: null | number,
    is_leaf: boolean,
    dtype: string,
    id: number,
    layout: { shape: Array<number>, strides: Array<number> },
    fuse_group: undefined | Array<number>,
    fuse_group_id: number,
    inputs: Array<number>,
    output_id: [number, number], // currently not used
    ref_cnt: number, // currently not used
    requires_grad: boolean, // currently not used
    scope_id: number, // currently not used
    span_msg: string | null,
}

function determineBlockType(input: string | { [key: string]: any }): BlockType {
    if (typeof input === 'string') {
        switch (input) {
            case 'Function':
                return BlockType.Function;
            case 'WhileCond':
                return BlockType.WhileCond;
            case "While":
                return BlockType.While;
            default:
                return BlockType.Unknown;
        }
    }

    if (typeof input === 'object' && input !== null) {
        if ('WhileBody' in input) {
            return BlockType.WhileBody;
        }
    }

    return BlockType.Unknown;
}

export function generate_custom_op_blocks(graph: any, blocks_manager: any): [{ data: BlockNode }[], Map<number, string>, Set<string>] {
    let block_set = new Set<number>(); // to avoid generate same block
    let block_ids = new Map<number, string>(); // to avoid generate same random id
    let random_id_set = new Set<string>();
    let cy_nodes = [];
    // graph is a dictionary of nodes
    // each node has a block_info which contains block ids, {current_id: number, parent_id: number}
    // the main block id is 0, any other block id > 0 is a child block of the main block

    // this step is to generate a random id for each block
    for (const block of Object.keys(blocks_manager.block_names)) {
        let block_id = Number(block);
        if (block_id > 0 && !block_set.has(block_id)) {
            let block_random_id: string = generateUniqueId(random_id_set);
            block_set.add(block_id);
            block_ids.set(block_id, block_random_id);
        }
    }
    block_set.clear();
    for (const node_id of Object.keys(graph)) {
        let node: RustNode = graph[node_id];
        let block_id: number = node.block_id;
        if (block_id > 0 && !block_set.has(block_id)) {
            block_set.add(block_id);
            let block_type = determineBlockType(blocks_manager.block_type[block_id]);
            let cy_node: BlockNode = {
                id: block_ids.get(block_id) || 'unknown',
                label: blocks_manager.block_names[block_id],
                block_id: node.block_id,
                type: NodeType.Block,
                block_type: block_type,
                expanded: false,
                inputs: new Set<string>(),
                outputs: new Set<string>(),
                preds: new Set<string>(),
            };
            cy_nodes.push({
                data: cy_node
            });
        }
    }

    // generate blocks that are not show in the graph, which is empty blocks
    for (const id of Object.keys(blocks_manager.block_names)) {
        let id_num: number = Number(id);
        if (!block_set.has(id_num)) {
            let block_type = determineBlockType(blocks_manager.block_type[id_num]);
            let cy_node: BlockNode = {
                id: block_ids.get(id_num) || 'unknown',
                label: blocks_manager.block_names[id_num],
                block_id: id_num,
                type: NodeType.Block,
                block_type: block_type,
                expanded: false,
                inputs: new Set<string>(),
                outputs: new Set<string>(),
                preds: new Set<string>(),
            }
            cy_nodes.push({
                data: cy_node
            });
        }
    }

    return [cy_nodes, block_ids, random_id_set];
}

export function generate_primitive_nodes(graph: any): any[] {
    // graph is a dictionary of nodes

    let cy_nodes = [];

    for (const node_id of Object.keys(graph)) {
        let node: RustNode = graph[node_id];
        let cy_node: PrimitiveNode = {
            id: node_id.toString(),
            label: node.const_val === null ? '' : `const(${node.const_val})`,
            is_const: node.const_val !== null,
            inputs: node.inputs.map((input) => input.toString()),
            layout: { shape: node.layout.shape, strides: node.layout.strides },
            block_id: node.block_id,
            dtype: node.dtype,
            type: NodeType.Primitive,
        }
        cy_node.fuse_group = node.fuse_group;
        cy_node.group_id = node.fuse_group_id;
        cy_nodes.push({
            data: cy_node
        });
    }
    return cy_nodes;
}

export function generate_op_nodes(graph: any, random_id_set: Set<string>): { data: OpNode }[] {
    // graph is a dictionary of nodes
    let cy_nodes = [];

    for (const node_id of Object.keys(graph)) {
        let node = graph[node_id];
        let op_id = generateUniqueId(random_id_set);
        let op_name = to_op_str(node.op);
        if (op_name === "Null") {
            continue;
        }
        let cy_node: OpNode = {
            id: op_id,
            label: op_name,
            inputs: node.inputs.map((input: any) => input.toString()),
            result_id: node_id,
            block_id: node.block_id,
            type: NodeType.Op,
            span_msg: node.span_msg,
        }
        cy_nodes.push({
            data: cy_node
        });
    }
    return cy_nodes;
}

export function connect_nodes(op_nodes: { data: OpNode }[]) {
    let edges = [];
    for (const node of op_nodes) {
        for (const input of node.data.inputs) {
            edges.push({
                group: "edges",
                data: {
                    source: input,
                    target: node.data.id,
                }
            });
        }
        let result_id = node.data.result_id;
        edges.push({
            group: "edges",
            data: {
                source: node.data.id.toString(),
                target: result_id.toString(),
            }
        });
    }
    return edges;
}

export function connect_block_nodes(cy: cytoscape.Core, blocks: { data: BlockNode }[]) {
    for (const block of blocks) {
        if (block.data.block_type == BlockType.WhileBody) {
            let parent = block.data.parent;
            if (parent === undefined) {
                throw new Error('Parent node not found, block_id: ' + block.data.block_id.toString());
            }
            cy.$id(parent).children().forEach(element => {
                let data = element.data();
                if (data.type === NodeType.Block && data.block_type === BlockType.WhileCond) {
                    block.data.preds?.add(data.id);
                    cy.add({
                        group: "edges",
                        data: {
                            source: data.id,
                            target: block.data.id,
                        }
                    });
                }
            });
        } else if (block.data.block_type == BlockType.WhileCond) {
            let parent = block.data.parent;
            if (parent === undefined) {
                throw new Error('Parent node not found, block_id: ' + block.data.block_id.toString());
            }
            cy.$id(parent).children().forEach(element => {
                let data = element.data();
                if (data.type === NodeType.Block && data.block_type === BlockType.WhileBody) {
                    block.data.preds?.add(data.id);
                    cy.add({
                        group: "edges",
                        data: {
                            source: data.id,
                            target: block.data.id,
                        }
                    });
                }
            });
        }
    }
}

export function make_compound_nodes(
    block_id: number,
    catogorize_nodes: Map<number, any>,
    block_ids: Map<number, string>,
    parent_graph: Map<number, number>
) {
    let nodes = catogorize_nodes.get(block_id);
    if (nodes === undefined) {
        return;
    }
    if (block_id === 0) {
        return;
    }
    let block_random_id = block_ids.get(block_id);
    for (const node of nodes) {
        if (node.data.type === NodeType.Block) {
            let parent_block_id = parent_graph.get(node.data.block_id);
            if (parent_block_id === undefined) {
                throw new Error('Parent block id not found, block_id: ' + node.data.block_id.toString());
            }
            let parent_block_random_id = block_ids.get(parent_block_id);
            if (parent_block_id !== 0 && parent_block_random_id === undefined) {
                throw new Error('Parent block random id not found, block_id: ' + parent_block_id.toString());
            }
            node.data.parent = parent_block_random_id;
            continue;
        }
        node.data.parent = block_random_id;
    }
}

export function set_block_entry(
    parents_graph: Map<number, Set<number>>,
    nodes_map: Map<any, any>,
    block_ids: Map<number, string>,
    nodes: any[]
) {
    for (const node of nodes) {
        if (node.data.type === NodeType.Op) {
            let current_block = node.data.block_id;
            let parents = parents_graph.get(current_block);
            if (parents === undefined) {
                parents = new Set<number>();
            }
            parents.add(current_block);
            let parents_array = Array.from(parents).sort((a, b) => b - a);
            parents.delete(current_block);
            let left = node.data.left;
            let right = node.data.right;
            if (left) {
                // determine whether the left node is in the same block or is the same block's child block
                let left_node = nodes_map.get(left);
                if (left_node === undefined) {
                    throw new Error('Left node not found, node_id: ' + left.toString());
                }
                let left_block = left_node.data.block_id;
                let common_block = find_common_block(parents_graph, current_block, left_block);
                // op node down to the common block, add input to all the parent blocks tuil the common block
                for (const parent_block of parents_array) {
                    if (parent_block === common_block || parent_block === 0) {
                        break;
                    }
                    let parent_block_random_id = block_ids.get(parent_block);
                    if (parent_block_random_id === undefined) {
                        throw new Error('Parent block random id not found, block_id: ' + parent_block.toString());
                    }
                    let parent_block_node = nodes_map.get(parent_block_random_id);
                    if (parent_block_node === undefined) {
                        throw new Error('Parent block node not found, block_id: ' + parent_block_random_id);
                    }
                    if (left_node.data.type !== NodeType.Primitive) {
                        throw new Error('Left node is not a primitive node, node_id: ' + left.toString());
                    }
                    parent_block_node.data.inputs.add(`${left_node.data.id}-${node.data.id}`);
                }
                let left_parents = parents_graph.get(left_block);
                if (left_parents !== undefined) {
                    left_parents.add(left_block);
                    let left_parents_array = Array.from(left_parents).sort((a, b) => b - a);
                    left_parents.delete(left_block);
                    // left node down to the common block, add ouput to all the parent blocks tuil the common block
                    for (const parent_block of left_parents_array) {
                        if (parent_block === common_block || parent_block === 0) {
                            break;
                        }
                        let parent_block_random_id = block_ids.get(parent_block);
                        if (parent_block_random_id === undefined) {
                            throw new Error('Parent block random id not found, block_id: ' + parent_block.toString());
                        }
                        let parent_block_node = nodes_map.get(parent_block_random_id);
                        if (parent_block_node === undefined) {
                            throw new Error('Parent block node not found, block_id: ' + parent_block_random_id);
                        }
                        parent_block_node.data.outputs.add(`${left_node.data.id}-${node.data.id}`);
                    }
                }
            }
            if (right) {
                // determine whether the right node is in the same block or is the same block's child block
                let right_node = nodes_map.get(right);
                if (right_node === undefined) {
                    throw new Error('Right node not found, node_id: ' + right.toString());
                }
                let right_block = right_node.data.block_id;
                let common_block = find_common_block(parents_graph, current_block, right_block);
                // op node down to the common block, add input to all the parent blocks tuil the common block
                for (const parent_block of parents_array) {
                    if (parent_block === common_block || parent_block === 0) {
                        break;
                    }
                    let parent_block_random_id = block_ids.get(parent_block);
                    if (parent_block_random_id === undefined) {
                        throw new Error('Parent block random id not found, block_id: ' + parent_block.toString());
                    }
                    let parent_block_node = nodes_map.get(parent_block_random_id);
                    if (parent_block_node === undefined) {
                        throw new Error('Parent block node not found, block_id: ' + parent_block_random_id);
                    }
                    if (right_node.data.type !== NodeType.Primitive) {
                        throw new Error('Right node is not a primitive node, node_id: ' + right.toString());
                    }
                    parent_block_node.data.inputs.add(`${right_node.data.id}-${node.data.id}`);
                }
                let right_parents = parents_graph.get(right_block);
                if (right_parents !== undefined) {
                    right_parents.add(right_block);
                    let right_parents_array = Array.from(right_parents).sort((a, b) => b - a);
                    right_parents.delete(right_block);
                    // left node down to the common block, add ouput to all the parent blocks tuil the common block
                    for (const parent_block of right_parents_array) {
                        if (parent_block === common_block || parent_block === 0) {
                            break;
                        }
                        let parent_block_random_id = block_ids.get(parent_block);
                        if (parent_block_random_id === undefined) {
                            throw new Error('Parent block random id not found, block_id: ' + parent_block.toString());
                        }
                        let parent_block_node = nodes_map.get(parent_block_random_id);
                        if (parent_block_node === undefined) {
                            throw new Error('Parent block node not found, block_id: ' + parent_block_random_id);
                        }
                        parent_block_node.data.outputs.add(`${right_node.data.id}-${node.data.id}`);
                    }
                }
            }
        }
    }
}

export function find_common_block(parent_blocks: Map<number, Set<number>>, a_block: number, b_block: number): number {
    let a_parents = parent_blocks.get(a_block);
    let b_parents = parent_blocks.get(b_block);
    if (a_parents === undefined || b_parents === undefined) {
        if (a_block === 0 || b_block === 0) {
            return 0;
        } else {
            throw new Error('Parent blocks not found, a_block: ' + a_block + ', b_block: ' + b_block);
        }
    }
    a_parents.add(a_block);
    b_parents.add(b_block);
    let common_blocks = [];
    for (const parent of a_parents) {
        if (b_parents.has(parent)) {
            common_blocks.push(parent);
        }
    }
    a_parents.delete(a_block);
    b_parents.delete(b_block);
    return Math.max(...common_blocks);
}

// function to catogorize nodes based on the block id
export function catogorize_nodes(nodes: any[], block_ids: Map<number, string>): Map<number, any> {
    let map: Map<number, any> = new Map();
    map.set(0, []);
    for (const block of block_ids.keys()) {
        map.set(Number(block), [])
    }
    // make blocks
    for (const node of nodes) {
        let block_id = node.data.block_id;
        map.get(block_id)?.push(node);
    }
    return map;
}

// function to catogorize cytoscape nodes based on the block id
export function catogorize_cy_nodes(nodes: any[], block_ids: Map<number, string>): Map<number, any> {
    let map: Map<number, any> = new Map();
    map.set(0, []);
    for (const block of block_ids.keys()) {
        map.set(Number(block), [])
    }
    for (const node of nodes) {
        let block_id = node.data('block_id');
        map.get(block_id).push(node);
    }
    return map;
}

export function create_parent_graph(graph: any): Map<number, number> {
    let parent_graph = new Map<number, number>();
    for (const node_id of Object.keys(graph)) {
        let id = Number(node_id);
        let parent = graph[id];
        parent_graph.set(id, parent);
    }
    return parent_graph;
}

export function create_parents_graph(graph: any): Map<number, Set<number>> {
    let parents_graph = new Map<number, Set<number>>();

    for (const node_id of Object.keys(graph)) {
        let id = Number(node_id);
        let array = graph[id];
        let parents = new Set<number>();
        for (const parent of array) {
            parents.add(parent);
        }
        parents_graph.set(id, parents);
    }
    return parents_graph;
}

export function create_child_graph(graph: any): Map<number, Set<number>> {
    let parents_graph = new Map<number, Set<number>>();

    for (const node_id of Object.keys(graph)) {
        let id = Number(node_id);
        let array = graph[id];
        let parents = new Set<number>();
        for (const parent of array) {
            parents.add(parent);
        }
        parents_graph.set(id, parents);
    }
    return parents_graph;
}

export function get_parent_block_ids(block_id: number, parent_block_ids: Map<number, number>) {
    let parent_block_id = parent_block_ids.get(block_id);
    let parent_block_ids_set = new Set<number>();
    while (parent_block_id !== undefined && !(block_id === 0 && parent_block_id === 0)) {
        parent_block_ids_set.add(parent_block_id);
        block_id = parent_block_id;
        parent_block_id = parent_block_ids.get(parent_block_id);
    }
    return parent_block_ids_set;
}

export function collapse(
    target_block: BlockNode,
    cy: cytoscape.Core,
    block_collapse_edges: Map<number, Set<cytoscape.CollectionReturnValue>>,
    block_collapse_nodes: Map<number, Set<cytoscape.CollectionReturnValue>>

) {
    let block_node = target_block;

    let descendants = cy.$id(block_node.id).descendants();
    let removed_nodes = new Set<cytoscape.CollectionReturnValue>();
    let removed_edges = new Set<cytoscape.CollectionReturnValue>();
    // remove all the inner edges
    descendants.forEach((child) => {
        child.incomers().forEach((ele) => {
            if (ele.isEdge()) {
                let source = ele.data('source');
                if (descendants.has(cy.$id(source))) {
                    // source is going to be collapsed, source is pointing to another node which is going to be collapsed
                    removed_edges.add(ele.remove());
                } else {
                    let removed = ele.remove();
                    let removed_data = removed.data();
                    if (removed_data.origin_target !== undefined && removed_data.origin_source !== undefined) {
                        removed_edges.add(removed);
                        let edge = cy.add({
                            group: "edges", data: {
                                source: source,
                                target: block_node.id,
                                origin_target: removed_data.origin_target,
                                origin_source: removed_data.origin_source
                            }
                        });
                        cy.add(edge);
                    } else {
                        removed.data('origin_target', ele.data('target'));
                        removed.data('origin_source', ele.data('source'));
                        let edge = cy.add({
                            group: "edges", data: {
                                source: source,
                                target: block_node.id,
                                origin_target: ele.data('target'),
                                origin_source: ele.data('source')
                            }
                        });
                        cy.add(edge);
                    }
                    removed_edges.add(removed);
                }
            }
        });
        child.outgoers().forEach((ele) => {
            if (ele.isEdge()) {
                let target = ele.data('target');
                if (descendants.has(cy.$id(target))) {
                    // target is from outside or is being collapsed
                    removed_edges.add(ele.remove());
                } else {
                    let removed = ele.remove();
                    let removed_data = removed.data();
                    if (removed_data.origin_target !== undefined && removed_data.origin_source !== undefined) {
                        removed_edges.add(removed);
                        let edge = cy.add({
                            group: "edges", data: {
                                source: block_node.id,
                                target: target,
                                origin_target: removed_data.origin_target,
                                origin_source: removed_data.origin_source
                            }
                        });
                        cy.add(edge);
                    } else {
                        removed.data('origin_target', ele.data('target'));
                        removed.data('origin_source', ele.data('source'));
                        let edge = cy.add({
                            group: "edges", data: {
                                source: block_node.id,
                                target: target,
                                origin_target: ele.data('target'),
                                origin_source: ele.data('source')
                            }
                        });
                        cy.add(edge);
                    }
                    removed_edges.add(removed);
                }
            }
        });
    });
    descendants.forEach((child) => {
        removed_nodes.add(child.remove());
    });
    if (!block_collapse_edges.has(block_node.block_id)) {
        block_collapse_edges.set(block_node.block_id, removed_edges);
    }
    if (!block_collapse_nodes.has(block_node.block_id)) {
        block_collapse_nodes.set(block_node.block_id, removed_nodes);
    }
    target_block.expanded = false;
}

export function get_sorted_parents(nodes_map: Map<any, any>, node_id: any, parent_graph: Map<number, number>): number[] {
    let stored_node_block = nodes_map.get(node_id).data.block_id;
    let parents = get_parent_block_ids(stored_node_block, parent_graph);
    parents.add(stored_node_block);
    let sorted_parents = Array.from(parents).sort((a, b) => b - a);
    return sorted_parents;
}

export function expand(
    target_block: any,
    cy: cytoscape.Core,
    nodes_map: Map<any, any>,
    block_ids: Map<number, string>,
    parent_graph: Map<number, number>,
    block_collapse_edges: Map<number, Set<cytoscape.CollectionReturnValue>>,
    block_collapse_nodes: Map<number, Set<cytoscape.CollectionReturnValue>>
) {
    let block_id = target_block.data().block_id;
    let removed_edges = block_collapse_edges.get(block_id);
    let removed_nodes = block_collapse_nodes.get(block_id);

    removed_nodes?.forEach((ele) => {
        cy.add(ele);
    });
    removed_edges?.forEach((ele) => {
        let source = ele.data('source');
        let target = ele.data('target');
        if (ele.data('origin_target') !== undefined && ele.data('origin_source') !== undefined) {
            source = ele.data('origin_source');
            target = ele.data('origin_target');
        }
        if (cy.$id(source).data() === undefined && cy.$id(target).data() !== undefined) {
            // souce is being collapsed
            let sorted_parents = get_sorted_parents(nodes_map, source, parent_graph);
            for (const parent of sorted_parents) {
                let block_random_id = block_ids.get(parent);
                if (block_random_id === undefined) {
                    continue;
                }
                let block = cy.$id(block_random_id).data();
                if (block == undefined) {
                    continue;
                } else {
                    cy.add({ group: "edges", data: { source: block.id, target: target } });
                    break;
                }
            }
        } else if (cy.$id(target).data() === undefined && cy.$id(source).data() !== undefined) {
            // target is being collapsed
            let sorted_parents = get_sorted_parents(nodes_map, target, parent_graph);
            for (const parent of sorted_parents) {
                let block_random_id = block_ids.get(parent);
                if (block_random_id === undefined) {
                    continue;
                }
                let block = cy.$id(block_random_id).data();
                if (block == undefined) {
                    continue;
                } else {
                    cy.add({ group: "edges", data: { source: source, target: block.id } });
                    break;
                }
            }
        } else if (cy.$id(target).data() === undefined && cy.$id(source).data() === undefined) {
            let new_source = source;
            let new_target = target;
            let sorted_parents = get_sorted_parents(nodes_map, source, parent_graph);
            for (const parent of sorted_parents) {
                let block_random_id = block_ids.get(parent);
                if (block_random_id === undefined) {
                    continue;
                }
                let block = cy.$id(block_random_id).data();
                if (block == undefined) {
                    continue;
                } else {
                    new_source = block.id;
                    break;
                }
            }
            let sorted_parents2 = get_sorted_parents(nodes_map, target, parent_graph);
            for (const parent of sorted_parents2) {
                let block_random_id = block_ids.get(parent);
                if (block_random_id === undefined) {
                    continue;
                }
                let block = cy.$id(block_random_id).data();
                if (block == undefined) {
                    continue;
                } else {
                    new_target = block.id;
                    break;
                }
            }
            cy.add({ group: "edges", data: { source: new_source, target: new_target } });
        } else {
            cy.add({ group: "edges", data: { source: source, target: target } })
        }
    });
    block_collapse_edges.delete(block_id);
    block_collapse_nodes.delete(block_id);

    // remove incomers
    target_block.incomers().forEach((ele: any) => {
        if (ele.isEdge()) {
            if (target_block.data().block_type === BlockType.WhileBody && target_block.data().preds.has(ele.data().source)) {
            } else {
                ele.remove();
            }
        }
    });
    // remove outgoers
    target_block.outgoers().forEach((ele: any) => {
        if (ele.isEdge()) {
            if (target_block.data().block_type === BlockType.WhileCond && target_block.data().preds.has(ele.data().target)) {
            } else {
                ele.remove();
            }
        }
    });

    target_block.data().expanded = true;
}