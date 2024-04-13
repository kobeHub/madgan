import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle


def create_swat_graph():
    # Create a new directed graph
    G = nx.DiGraph()

    # Define the stages
    stages = [
        ['Raw Water', 'MV101', 'Raw Water Tank(LIT101)', 'Pump(p101)'],
        ['P201', 'P203', 'P205', 'Static Mixer', 'MV201'],
        ['UF Feed Tank(LIT301)', 'UF Feed Pump(P301)', 'UF(T301)'],
        ['RO Feed Tank(LIT401)', 'RO Feed Pump(P401)', 'UV'],
        ['Cartridge Filter', 'RO Boost Pump (P501)', 'RO Unit'],
        ['UF Permeate Tank', 'Water Recycled'],
        ['UF Backwash Tank', 'UF Backwash Pump(P602)'],
    ]

    # Add nodes and edges for each stage
    for i, stage in enumerate(stages):
        for j in range(len(stage) - 1):
            G.add_edge(stage[j], stage[j + 1])
        # Add edge from the last element of the current stage to the first element of the next stage
        if i < len(stages) - 2:
            G.add_edge(stage[-1], stages[i + 1][0])

    G.add_edge(stages[-3][-1], stages[-1][0])
    G.add_edge(stages[-1][-1], stages[2][-1])

    return G, stages


def display_swat_graph():
    G, stages = create_swat_graph()

    # pos = nx.spring_layout(G, k=0.6)
    pos = nx.kamada_kawai_layout(G, scale=10)
    plt.figure(figsize=(10, 10))

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue')

    # Draw the edges
    nx.draw_networkx_edges(G, pos)
    # Draw the edges with arrows
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15)
    # Draw the node labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold')
    # Draw rectangles around each stage
    ax = plt.gca()
    margin = 0.15  # adjust this value to change the size of the margin
    for stage in stages:
        stage_pos = [pos[node] for node in stage]
        min_x = min(pos[0] for pos in stage_pos) - margin
        min_y = min(pos[1] for pos in stage_pos) - margin
        max_x = max(pos[0] for pos in stage_pos) + margin
        max_y = max(pos[1] for pos in stage_pos) + margin
        rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    with open('models/graph.pkl', 'wb') as f:
        pickle.dump(G, f)

    # Save the figure to a file
    plt.savefig("img/swat-graph-model.png", dpi=300)
