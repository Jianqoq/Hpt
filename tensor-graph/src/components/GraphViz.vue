<template>
    <div id="graph-container"></div>
    <div id="button-container">
        <button @click="get_hir_graph">Hir</button>
        <button @click="get_mir_graph">Mir</button>
        <button @click="get_raw_graph">Raw</button>
    </div>
    <transition name="fade">
        <div v-if="showErrorTooltip" class="error-tooltip" :style="tooltipStyle">
            {{ errorMessage }}
        </div>
    </transition>
</template>

<script>
import { invoke } from '@tauri-apps/api/tauri';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import { getTextWidthHeight, createCy, generateUniqueId, format_primitive, format_mir, html_label } from '../components/GraphUtils';
import contextMenus from 'cytoscape-context-menus';
import 'cytoscape-context-menus/cytoscape-context-menus.css';
import '@fortawesome/fontawesome-free/css/all.min.css';
import nodeHtmlLabel from 'cytoscape-node-html-label-custom';

import {
    catogorize_nodes, connect_nodes,
    create_parent_graph, create_child_graph,
    generate_custom_op_blocks, generate_op_nodes,
    generate_primitive_nodes, make_compound_nodes,
    catogorize_cy_nodes,
    NodeType,
    create_parents_graph,
    set_block_entry,
    collapse,
    get_sorted_parents,
    expand,
    connect_block_nodes,
} from '../components/nodes_gen/nodes_gen';
import { BlockType } from './nodes_gen/nodes_gen';
cytoscape.use(dagre);
cytoscape.use(contextMenus);
cytoscape.use(nodeHtmlLabel);

export default {

    data() {
        return {
            showErrorTooltip: false,
            errorMessage: '',
            tooltipStyle: {
                position: 'fixed',
                left: '0px',
                top: '0px'
            }
        };
    },

    mounted() {
        this.get_raw_graph();
    },
    methods: {

        async get_raw_graph() {
            try {
                const computation_graph = await invoke('get_graph_data');
                console.log(computation_graph);
                this.loadGraphData(computation_graph, format_primitive, "raw");
            } catch (error) {
                console.error('Failed to load graph data:', error);
            }
        },

        async get_hir_graph() {
            try {
                const computation_graph = await invoke('get_hir_data');
                this.loadGraphData(computation_graph);
            } catch (error) {
                console.error('Failed to load graph data:', error);
            }
        },

        async get_mir_graph() {
            try {
                const computation_graph = await invoke('get_mir_data');
                console.log(computation_graph);
                let mir_menu = [
                    {
                        id: 'zip instruction',
                        content: 'zip instruction',
                        tooltipText: 'zip instruction',
                        selector: "node[type = 1]",
                        onClickFunction: function (event) {
                            let node = computation_graph.nodes[event.target.data().id];
                            console.log(node.id + ': ' + node.zip_instructions);
                        },
                    },
                ]

                this.loadGraphData(computation_graph, format_mir, "mir", mir_menu);
            } catch (error) {
                console.error('Failed to load graph data:', error);
            }
        },

        default_menu(
            cy,
            nodes_map,
            block_ids,
            parent_graph,
            catogorized_cy_nodes,
            child_graphs,
            block_collapse_edges,
            block_collapse_nodes,
            format_method,
            lis
        ) {
            return [
                {
                    id: 'collapse',
                    content: 'collapse',
                    tooltipText: 'collapse',
                    selector: "node[type = 0][?expanded]",
                    onClickFunction: function (event) {
                        let block_node = event.target.data();
                        cy.batch(
                            () => {
                                collapse(
                                    block_node,
                                    cy,
                                    block_collapse_edges,
                                    block_collapse_nodes
                                );
                                format_method(cy);
                            }
                        );
                    },
                    hasTrailingDivider: true
                },
                {
                    id: 'expand',
                    content: 'expand',
                    tooltipText: 'expand',
                    selector: 'node[type = 0][!expanded]',
                    onClickFunction: function (event) {
                        let block_id = event.target.data('block_id');
                        if (block_id == 0) {
                            return;
                        }
                        cy.on('style', lis);
                        cy.batch(
                            () => {
                                expand(
                                    event.target,
                                    cy,
                                    nodes_map,
                                    block_ids,
                                    parent_graph,
                                    block_collapse_edges,
                                    block_collapse_nodes
                                );
                                format_method(cy);
                            }
                        );
                        cy.off('style', lis);
                    }
                },
                {
                    id: 'layout',
                    content: 'layout',
                    tooltipText: 'layout',
                    selector: 'node[type = 1]',
                    onClickFunction: function (event) {
                        let node = nodes_map.get(event.target.data().id);
                        console.log(
                            node.data.id + ': ' + 'shape: [' + node.data.layout.shape + '], strides: [' + node.data.layout.strides + ']'
                        );
                    }
                }
            ]
        },

        async loadGraphData(computation_graph, format_method, mode, special_menu = null) {
            console.log(computation_graph);
            const nodes = [];
            let op_ids = new Set();
            let has_error = false;
            let parent_graph = create_parent_graph(computation_graph.blocks_manager.block_parent);
            let child_graphs = create_child_graph(computation_graph.blocks_manager.block_children);
            let parents_graph = create_parents_graph(computation_graph.blocks_manager.block_parents);
            let [block_nodes, block_ids, random_id_set] = generate_custom_op_blocks(computation_graph.nodes, computation_graph.blocks_manager);
            let primitive_nodes = generate_primitive_nodes(computation_graph.nodes);
            let op_nodes = generate_op_nodes(computation_graph.nodes, random_id_set);
            nodes.push(...block_nodes);
            nodes.push(...primitive_nodes);
            nodes.push(...op_nodes);
            let nodes_map = new Map();
            for (const node of nodes) {
                nodes_map.set(node.data.id, node);
            }
            set_block_entry(parents_graph, nodes_map, block_ids, nodes);

            let edges = connect_nodes(op_nodes);
            let catogorized_nodes = catogorize_nodes(nodes, block_ids);
            for (let i = 1; i <= block_ids.size; i++) {
                make_compound_nodes(i, catogorized_nodes, block_ids, parent_graph);
            }

            let block_collapse_edges = new Map();
            let block_collapse_nodes = new Map();
            let cy = createCy(nodes, edges, "graph-container");
            let block_edges = connect_block_nodes(cy, block_nodes);
            let catogorized_cy_nodes = catogorize_cy_nodes(cy.nodes(), block_ids);

            let node_ref_cnts = new Map();
            for (const [child, parent] of parent_graph) {
                node_ref_cnts.set(parent, node_ref_cnts.get(parent) + 1 || 1);
            }

            // collapse all nodes recursively, it is for initialization
            cy.batch(() => {
                cy.nodes().forEach((node) => {
                    let type = node.data("type");
                    if (type == NodeType.Block) {
                        let children = child_graphs.get(node.data().block_id);
                        if (children == undefined) {
                            // this is a leaf node, recursively find its parent and collapse them
                            let parents = parents_graph.get(node.data().block_id);
                            parents.add(node.data().block_id);
                            let sorted_parents = Array.from(parents).sort((a, b) => b - a);
                            node_ref_cnts.set(node.data().block_id, 1);
                            parents.delete(node.data().block_id);
                            for (const parent of sorted_parents) {
                                let ref_cnt = node_ref_cnts.get(parent);
                                if (ref_cnt - 1 == 0) {
                                    if (parent == 0) {
                                        break;
                                    }
                                    let parent_id = block_ids.get(parent);
                                    collapse(
                                        cy.$id(parent_id).data(),
                                        cy,
                                        block_collapse_edges,
                                        block_collapse_nodes
                                    );
                                    node_ref_cnts.set(parent, ref_cnt - 1);
                                } else {
                                    let parent_id = block_ids.get(parent);
                                    node_ref_cnts.set(parent, ref_cnt - 1);
                                    break;
                                }
                            }
                        }
                    }
                });
            });
            let a = html_label(cy, mode);
            format_method(cy);
            let default_menu = this.default_menu(
                cy,
                nodes_map,
                block_ids,
                parent_graph,
                catogorized_cy_nodes,
                child_graphs,
                block_collapse_edges,
                block_collapse_nodes,
                format_method,
                a.updateDataOrStyleCyHandler
            );
            if (special_menu) {
                default_menu = default_menu.concat(special_menu);
            }
            cy.contextMenus({
                menuItems: default_menu
            });
            cy.on('mouseover', 'node[type=2][?span_msg]', (event) => {
                this.errorMessage = '';
                let node = event.target;
                let nodePosition = node.renderedPosition();
                let nodeHeight = node.renderedHeight();
                let nodeWidth = node.renderedWidth();
                this.showErrorTooltip = true;
                let span_msg = node.data().span_msg;
                let max_len = 0;
                let rows = 0;
                for (const [key, value] of Object.entries(span_msg)) {
                    let msg = `${key}: ${value}\n`;
                    let text_width = getTextWidthHeight(msg, '12px Arial');
                    max_len = Math.max(max_len, text_width[0]);
                    this.errorMessage += msg;
                    rows++;
                }
                let exceed_width = (max_len - nodeWidth) / 2;
                this.tooltipStyle.left = `${nodePosition.x - nodeWidth / 2 - exceed_width - 8 - 5}px`;
                this.tooltipStyle.top = `${nodePosition.y - nodeHeight / 2 - rows * 10}px`;
            });
            cy.on('mouseout', 'node[type=2][?span_msg]', () => {
                this.showErrorTooltip = false;
            });
        },
    }
};
</script>

<style scoped>
#graph-container {
    width: 100%;
    height: 100%;
}

.error-tooltip {
  background-color: #ffdddd;
  border: 1px solid #ff0000;
  padding: 8px;
  border-radius: 5px;
  box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
  white-space: pre-wrap;
  z-index: 10;
}

.fade-enter-active, .fade-leave-active {
  transition: opacity 0.25s;
}

.fade-enter, .fade-leave-to {
  opacity: 0;
}

#button-container {
    position: absolute;
    top: 10px;
    right: 100px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
</style>