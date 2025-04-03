"""
CSC111 Project 2 main file
"""

from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
import csv
import os
import networkx as nx
import plotly.graph_objs as go
from graphviz import Digraph


# Data filtering part
def filter_data(input_file: str, output_file: str) -> None:
    """
    Filters <input_file> and saves as <output_file>
    :param input_file: The raw data file name to be filtered for main functions of program
    :param output_file: The name output file should be saved as
    """
    # If the output file already exists, return and don't do anything
    if os.path.exists(output_file):
        return

    # List of columns to keep
    columns_to_keep = ['ticker', 'asset_description', 'type', 'representative',
                       'cap_gains_over_200_usd', 'sector', 'party']

    # Open the input file and the output file
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)

        # Ensure the headers match the columns we want to keep
        fieldnames = [field for field in reader.fieldnames if field in columns_to_keep]

        # Open the output file to write filtered data
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            # Write the filtered rows
            for row in reader:
                filtered_row = {key: row[key] for key in fieldnames}
                writer.writerow(filtered_row)


# Graph part
class _Vertex:
    """A vertex in a graph.

    Instance Attributes:
        - item: The data stored in this vertex.
        - neighbours: The vertices that are adjacent to this vertex.

    Representation Invariants:
        - self not in self.neighbours
        - all(self in u.neighbours for u in self.neighbours)
    """
    item: Any
    kind: str
    neighbours: set[_Vertex]

    def __init__(self, item: Any, kind: str) -> None:
        """Initialize a new vertex with the given item and neighbours."""
        self.item = item
        self.kind = kind
        self.neighbours = set()

    def check_connected(self, target_item: Any, visited: set[_Vertex]) -> bool:
        """Return whether this vertex is connected to a vertex corresponding to the target_item,
        WITHOUT using any of the vertices in visited.

        Preconditions:
            - self not in visited
        """
        if self.item == target_item:
            # Our base case: the target_item is the current vertex
            return True
        else:
            visited.add(self)         # Add self to the set of visited vertices
            for u in self.neighbours:
                if u not in visited:  # Only recurse on vertices that haven't been visited
                    if u.check_connected(target_item, visited):
                        return True

            return False


class Graph:
    """A graph.

    Representation Invariants:
        - all(item == self._vertices[item].item for item in self._vertices)
    """
    # Private Instance Attributes:
    #     - _vertices:
    #         A collection of the vertices contained in this graph.
    #         Maps item to _Vertex object.
    _vertices: dict[Any, _Vertex]

    def __init__(self) -> None:
        """Initialize an empty graph (no vertices or edges)."""
        self._vertices = {}

    def add_vertex(self, item: Any, kind: str) -> None:
        """Add a vertex with the given item and kind to this graph.

        The new vertex is not adjacent to any other vertices.
        Do nothing if the given item is already in this graph.
        """
        if item not in self._vertices:
            self._vertices[item] = _Vertex(item, kind)

    def add_edge(self, item1: Any, item2: Any) -> None:
        """Add an edge between the two vertices with the given items in this graph.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.

        Preconditions:
            - item1 != item2
        """
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            v2 = self._vertices[item2]

            v1.neighbours.add(v2)
            v2.neighbours.add(v1)
        else:
            raise ValueError

    def adjacent(self, item1: Any, item2: Any) -> bool:
        """Return whether item1 and item2 are adjacent vertices in this graph.

        Return False if item1 or item2 do not appear as vertices in this graph.
        """
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            return any(v2.item == item2 for v2 in v1.neighbours)
        else:
            return False

    def connected(self, item1: Any, item2: Any) -> bool:
        """Return whether item1 and item2 are connected vertices in this graph.

        Return False if item1 or item2 do not appear as vertices in this graph.
        """
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            return v1.check_connected(item2, set())
        else:
            return False

    def get_all_vertices(self, kind: str = '') -> set:
        """Return a set of all vertex items in this graph.

        If kind != '', only return the items of the given vertex kind.
        """
        if kind != '':
            return {v.item for v in self._vertices.values() if v.kind == kind}
        else:
            return set(self._vertices.keys())

    def to_networkx(self, max_vertices: int = 5000) -> nx.Graph:
        """Convert this graph into a networkx Graph.

        max_vertices specifies the maximum number of vertices that can appear in the graph.
        (This is necessary to limit the visualization output for large graphs.)

        """
        graph_nx = nx.Graph()
        for v in self._vertices.values():
            graph_nx.add_node(v.item, kind=v.kind)

            for u in v.neighbours:
                if graph_nx.number_of_nodes() < max_vertices:
                    graph_nx.add_node(u.item, kind=u.kind)

                if u.item in graph_nx.nodes:
                    graph_nx.add_edge(v.item, u.item)

            if graph_nx.number_of_nodes() >= max_vertices:
                break

        return graph_nx


def load_graph_and_sectors(stock_data_file: str) -> Graph:
    """
    Loads congress stock trading data into graph
    """

    graph = Graph()

    with open(stock_data_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            graph.add_vertex(row[0], row[5])
            graph.add_vertex(row[3], row[6])

            graph.add_edge(row[3], row[0])

    return graph


def sector_clusters_list(graph: Graph, sectors: set[str]) -> list[set[str]]:
    """
    Returns a list of sets of verticies in the same sector
    """

    clusters_list = []
    for sector in sectors:
        cluster = graph.get_all_vertices(sector)
        clusters_list.append(cluster)

    return clusters_list


def get_layout_positions(nx_graph: nx.Graph, layout: str = 'spring_layout') -> dict[str, tuple[float, float]]:
    """
    Calculate and return the positions of nodes based on the chosen layout.
    """
    layout_func = getattr(nx, layout)
    return layout_func(nx_graph, seed=42, k=0.25, iterations=50)


def get_sector_to_color(nx_graph: nx.Graph) -> dict[str, str]:
    """
    Map the 'kind' attribute of each node to a color based on the sector.
    """
    stock_palette = [
        '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
        '#e377c2', '#bcbd22', '#17becf', '#8B008B',
        '#00ced1', '#FFD700', '#C71585'
    ]

    all_kinds = {nx_graph.nodes[node].get('kind', '') for node in nx_graph.nodes}
    party_aff = {'republican', 'democrat', 'independent'}

    stock_kinds = sorted(k for k in all_kinds if k.lower() not in party_aff and k.strip() != '')
    sector_to_color = {}

    for i, sk in enumerate(stock_kinds):
        sector_to_color[sk] = stock_palette[i % len(stock_palette)]

    return sector_to_color


def get_edges_trace(nx_graph: nx.Graph, pos: dict[str, tuple[float, float]]) -> go.Scatter:
    """
    Create the edge trace for Plotly visualization.
    """
    edge_x, edge_y = [], []

    for u, v in nx_graph.edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    return go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line={'color': 'rgba(180,180,180,0.4)', 'width': 0.5},
        hoverinfo='none'
    )


def get_node_properties(kind: str, deg: int, max_degree: int, node: str,
                        sector_to_color: dict[str, str]) -> tuple[str, float, str]:
    """
    Helper function to determine node properties (color, size, hover text)
    based on the 'kind' of the node.
    """
    kind_lower = kind.lower()

    if kind_lower == 'republican':
        color = '#FF0000'
        node_size = 5 + (deg / max_degree) * 5
        hover_text = f"{node}<br>Type: Politician<br>Party: Republican<br>Degree: {deg}"
    elif kind_lower == 'democrat':
        color = '#00008B'
        node_size = 5 + (deg / max_degree) * 5
        hover_text = f"{node}<br>Type: Politician<br>Party: Democrat<br>Degree: {deg}"
    elif kind_lower == 'independent':
        color = '#808080'
        node_size = 5 + (deg / max_degree) * 5
        hover_text = f"{node}<br>Type: Politician<br>Party: Independent<br>Degree: {deg}"
    else:
        color = sector_to_color.get(kind, '#999')
        node_size = 7 + (deg / max_degree) * 12
        hover_text = f"{node}<br>Type: Stock<br>Sector: {kind}<br>Degree: {deg}"

    return color, node_size, hover_text


def get_nodes_trace(nx_graph: nx.Graph, pos: dict[str, tuple[float, float]],
                    max_degree: int, sector_to_color: dict[str, str]) -> go.Scatter:
    """
    Create the node trace for Plotly visualization.
    """
    node_x, node_y, node_colors, node_sizes, hover_texts = [], [], [], [], []
    degrees = dict(nx_graph.degree)

    for node in nx_graph.nodes:
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])

        # Get properties for the current node using the helper function
        color, node_size, hover_text = get_node_properties(nx_graph.nodes[node].get('kind', ''), int(degrees[node]),
                                                           max_degree, node, sector_to_color)

        node_colors.append(color)
        node_sizes.append(node_size)
        hover_texts.append(hover_text)

    return go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker={'size': node_sizes, 'color': node_colors, 'line': {'width': 0.5, 'color': 'black'}},
        text=hover_texts,
        hovertemplate='%{text}',
        hoverlabel={'namelength': 0}
    )


def visualize_graph(graph: Graph, layout: str = 'spring_layout', max_nodes: int = 5000) -> None:
    """
    Visualize a graph where:
      - Politicians are labeled as 'Republican', 'Democrat', or 'Independent'.
      - All other vertices are treated as stocks, colored by sector.
    """
    # Convert graph to NetworkX format
    nx_graph = graph.to_networkx(max_nodes)

    # Get node positions based on the layout
    pos = get_layout_positions(nx_graph, layout)

    # Get sector to color mapping
    sector_to_color = get_sector_to_color(nx_graph)

    # Get edge trace
    edge_trace = get_edges_trace(nx_graph, pos)

    # Get node trace with node size, color, and hover text
    degrees = dict(nx_graph.degree)
    max_degree = max(degrees.values()) if degrees else 1
    node_trace = get_nodes_trace(nx_graph, pos, max_degree, sector_to_color)

    # Create the Plotly figure
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title='Congress Trading Graph',
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin={'l': 0, 'r': 0, 't': 40, 'b': 0},
        xaxis={'visible': False},
        yaxis={'visible': False}
    )

    # Show the plot
    fig.show()


# BST part
@dataclass
class BinarySearchTree:
    """
    Binary Search Tree class to visualize data
    """
    # Type annotations
    win_rate: float
    name: Optional[str] = None
    party: Optional[str] = None
    left: Optional['BinarySearchTree'] = None
    right: Optional['BinarySearchTree'] = None

    def __init__(self, win_rate: float, name: Optional[str] = None, party: Optional[str] = None) -> None:
        self.win_rate = win_rate
        self.name = name
        self.party = party
        self.left = None
        self.right = None


def insert(root: Optional[BinarySearchTree], win_rate: float, name: str, party: str) -> BinarySearchTree:
    """
    Function that determines how to insert data into BST depending on whether root is none or value is
    greater than / less than root
    """
    if root is None:
        return BinarySearchTree(win_rate, name, party)
    else:
        if win_rate <= root.win_rate:
            root.left = insert(root.left, win_rate, name, party)
        else:
            root.right = insert(root.right, win_rate, name, party)
        return root


def add_to_dot(node: Optional[BinarySearchTree], dot: Digraph, node_id_counter: list[int]) -> Optional[str]:
    """
    Function translating the BST into a visual graph using Digraph library
    """
    if node is None:
        return None

    node_id = f"node{node_id_counter[0]}"
    node_id_counter[0] += 1

    label = f"{node.name}: {node.win_rate:.2%}" if node.name != "ROOT" else "50%"
    tooltip = f"{node.name}" if node.name else ""

    # Set background color based on party
    color = "lightgray"
    if node.name != "ROOT":
        if node.party:
            if "Democrat" in node.party:
                color = "dodgerblue"
            elif "Republican" in node.party:
                color = "crimson"
        else:
            color = "purple"  # Third party

    dot.node(node_id, label=label, tooltip=tooltip, style='filled', fillcolor=color)

    has_left = node.left is not None
    has_right = node.right is not None

    # Add real children
    if has_left:
        left_id = add_to_dot(node.left, dot, node_id_counter)
        dot.edge(node_id, left_id)
    if has_right:
        right_id = add_to_dot(node.right, dot, node_id_counter)
        dot.edge(node_id, right_id)

    # Add invisible dummy node if only one child exists, to help with layout
    if has_left and not has_right:
        dummy_id = f"dummy{node_id_counter[0]}"
        node_id_counter[0] += 1
        dot.node(dummy_id, label="", style="invis")
        dot.edge(node_id, dummy_id, style="invis")
    elif has_right and not has_left:
        dummy_id = f"dummy{node_id_counter[0]}"
        node_id_counter[0] += 1
        dot.node(dummy_id, label="", style="invis")
        dot.edge(node_id, dummy_id, style="invis")

    return node_id


def load_data(filename: str) -> tuple[dict, dict]:
    """
    Load representative trade data from CSV.
    """
    rep_stats = {}  # Maps name to stats
    party_map = {}  # Maps names to political party
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            name = row[3]
            win = row[4].strip().upper() == 'TRUE'
            political_party = row[6]
            if name not in rep_stats:
                rep_stats[name] = {'true_count': 0, 'total': 0}
                party_map[name] = political_party
            rep_stats[name]['total'] += 1
            if win:
                rep_stats[name]['true_count'] += 1
    return rep_stats, party_map


def compute_win_rates(rep_stats: dict, party_map: dict) -> list[tuple[str, float, str]]:
    """
    Compute win rates for each representative.
    """
    rep_win_rates = []
    for name, stats in rep_stats.items():
        true_count = stats['true_count']
        total = stats['total']
        if true_count == 0:
            continue
        win_rate = true_count / total if total > 0 else 0
        rep_win_rates.append((name, win_rate, party_map[name]))
    return rep_win_rates


def build_balanced_bst(sorted_nodes: list) -> BinarySearchTree | None:
    """
    Recursively builds a balanced binary search tree (BST) from a sorted list of nodes.
    The nodes are tuples of (name, rate, party).
    """
    if not sorted_nodes:
        return None

    mid = len(sorted_nodes) // 2
    name, rate, party = sorted_nodes[mid]

    # Create a new node and recursively build left and right subtrees
    node = BinarySearchTree(rate, name, party)
    node.left = build_balanced_bst(sorted_nodes[:mid])
    node.right = build_balanced_bst(sorted_nodes[mid + 1:])

    return node


def sort_bst() -> Digraph:
    """
    The main function that organizes the data, sorts it, and builds a balanced BST.
    It also creates a DOT representation of the tree.
    """
    # Load data and compute win rates
    rep_stats, party_map = load_data('House_data.csv')
    rep_win_rates = compute_win_rates(rep_stats, party_map)

    dot = Digraph()  # Initialize the DOT graph
    node_id_counter = [0]  # Counter to assign unique IDs to nodes in the DOT graph

    # Separate the nodes based on win rate categories
    less_than = [(_name, _rate, _party) for _name, _rate, _party in rep_win_rates if _rate < 0.5]
    greater_than = [(_name, _rate, _party) for _name, _rate, _party in rep_win_rates if _rate > 0.5]
    equal = [(_name, _rate, _party) for _name, _rate, _party in rep_win_rates if _rate == 0.5]

    # Sort the lists by win rate
    less_than.sort(key=lambda x: x[1])
    greater_than.sort(key=lambda x: x[1])

    # Build the BST with 0.5 as the root
    root = BinarySearchTree(0.5, "ROOT", None)
    root.left = build_balanced_bst(less_than)
    root.right = build_balanced_bst(greater_than)

    # Insert equal-to-0.5 nodes into the BST
    for name, rate, party in equal:
        insert(root, rate, name, party)

    # Add the constructed tree to the DOT graph
    add_to_dot(root, dot, node_id_counter)

    return dot


def main() -> None:
    """
    Main function of the program that calls other functions
    """
    # Filters the raw date to needed data in right format
    filter_data('all_transactions.csv', 'House_data.csv')

    # Build win rate BST and render visualization.
    dot = sort_bst()
    dot.render('bst_winrate', view=True, format='png')

    # Makes the graph
    g = load_graph_and_sectors("House_data.csv")
    visualize_graph(g)


if __name__ == "__main__":
    main()

    # import python_ta.contracts
    #
    # python_ta.contracts.check_all_contracts()
    #
    # import doctest
    #
    # doctest.testmod()
    #
    # import python_ta
    #
    # python_ta.check_all(config={
    #     'max-line-length': 120,
    #     'disable': ['E1136'],
    #     'extra-imports': ['csv', 'networkx', 'os', 'graphviz', 'plotly.graph_objs'],
    #     'allowed-io': ['filter_data', 'load_graph_and_sectors', 'load_data'],
    #     'max-nested-blocks': 4
    # })
