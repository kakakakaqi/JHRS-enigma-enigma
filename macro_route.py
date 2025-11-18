from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from math import sqrt
import math
from typing import Any, Collection
import itertools
from typing import Any, Set, Tuple, List, FrozenSet, Collection, Dict, Optional
from collections import deque


@dataclass
class BareNode:
    room: Any
    paths: list[BareNode] = field(default_factory=list)


def import_rooms(rooms: list, root_idx):
    room = rooms[root_idx]
    root = BareNode(room)

    name2bare = {r.name: BareNode(r) for r in rooms}

    def f(node: BareNode):
        if len(node.paths) > 0:
            return
        node.paths = list(map(lambda x: name2bare[x.name], list(node.room.adjacent)))
        for node in node.paths:
            f(node)

    f(root)

    return root


def traverse_graph(
    start: BareNode,
) -> Tuple[Set[str], Set[FrozenSet[str]], Dict[str, BareNode]]:
    """
    Traverse graph using room.name as unique identifier.
    Returns (room_names, edges_by_name, name_to_node_map)
    """
    room_names = set()
    edges = set()
    name_to_node = {}

    stack = [start]
    visited_names = set()

    while stack:
        node = stack.pop()
        name = node.room.name

        if name not in visited_names:
            visited_names.add(name)
            room_names.add(name)
            name_to_node[name] = node

            for neighbor in node.paths:
                neighbor_name = neighbor.room.name
                # Store edge as unordered pair of room names
                edge = frozenset([name, neighbor_name])
                edges.add(edge)

                if neighbor_name not in visited_names:
                    stack.append(neighbor)

    return room_names, edges, name_to_node


def is_connected_by_name(
    start: BareNode, room_names: Set[str], name_to_node: Dict[str, BareNode]
) -> bool:
    """Check if graph is connected using room.name as identifier"""
    if not room_names:
        return True

    visited = set()
    queue = deque([start.room.name])

    while queue:
        name = queue.popleft()
        if name not in visited:
            visited.add(name)
            node = name_to_node[name]
            for neighbor in node.paths:
                neighbor_name = neighbor.room.name
                if neighbor_name not in visited:
                    queue.append(neighbor_name)

    return visited == room_names


def forms_tree_by_name(
    edge_set: Collection[FrozenSet[str]],
    room_names: Set[str],
    name_to_node: Dict[str, BareNode],
) -> bool:
    """
    Check if edge_set forms a valid spanning tree using room.name identifiers.
    Verifies: V-1 edges, fully connected, and acyclic.
    """
    V = len(room_names)
    if len(edge_set) != V - 1:
        return False

    if V <= 1:
        return True

    # Build adjacency list from edge_set
    adj = {name: set() for name in room_names}
    for edge in edge_set:
        u, v = tuple(edge)
        adj[u].add(v)
        adj[v].add(u)

    # DFS to check connectivity and detect cycles
    visited = set()

    def dfs(name: str, parent: Optional[str]) -> bool:
        visited.add(name)
        for neighbor_name in adj[name]:
            if neighbor_name == parent:
                continue
            if neighbor_name in visited:
                return False  # Back edge found = cycle
            if not dfs(neighbor_name, name):
                return False
        return True

    start_name = next(iter(room_names))
    return dfs(start_name, None) and len(visited) == V


def find_removable_paths(start: BareNode) -> List[List[Tuple[str, str]]]:
    """
    Find all combinations of paths whose removal makes the graph a tree.

    Returns:
        A list of combinations. Each combination is a list of edges to remove.
        Each edge is represented as a tuple of two room names: (room1, room2).
        Room names are sorted alphabetically within each tuple for consistency.
        Returns [[]] if the graph is already a tree.
        Returns [] if the graph is disconnected or cannot be converted.

    Example:
        For a triangle graph with rooms A, B, C:
        [[('A', 'B')], [('A', 'C')], [('B', 'C')]]
        This means you can remove edge AB, or remove edge AC, or remove edge BC.
    """
    if start is None:
        return []

    room_names, edges, name_to_node = traverse_graph(start)
    V, E = len(room_names), len(edges)

    # Graph must be connected
    if V == 0 or not is_connected_by_name(start, room_names, name_to_node):
        return []

    k = E - (V - 1)  # Number of edges to remove to get V-1 edges

    if k < 0:  # Not enough edges to form a tree
        return []

    if k == 0:  # Already has V-1 edges, verify it's a tree
        return [[]] if forms_tree_by_name(edges, room_names, name_to_node) else []

    # Find all combinations of k edges whose removal yields a tree
    removable_combinations = []
    for to_remove in itertools.combinations(edges, k):
        remaining = edges - set(to_remove)
        if forms_tree_by_name(remaining, room_names, name_to_node):
            # Convert frozenset of frozensets to list of sorted tuples
            combination = [tuple(sorted(edge)) for edge in to_remove]
            removable_combinations.append(combination)

    return removable_combinations


# Helper function to build graphs with circular references
def build_graph(adj_dict: dict) -> BareNode:
    """
    Build a BareNode graph from adjacency dictionary.
    adj_dict: {room_name: [neighbor_room_names]}
    Assumes room objects have a .name attribute.
    """
    # First pass: create empty node shells
    nodes = {}
    for name in adj_dict:
        # Create placeholder room object with name attribute
        nodes[name] = BareNode(type("Room", (), {"name": name})(), [])

    # Second pass: populate paths (bypassing frozen restriction)
    for name, neighbor_names in adj_dict.items():
        paths = tuple(nodes[n] for n in neighbor_names)
        object.__setattr__(nodes[name], "paths", paths)

    return next(iter(nodes.values()))


def construct_true_tree(rooms, root_idx, ignore_pairs):
    name2node_original = {
        r.name: Node(
            r.p,
            None,
            [],
            t=(0, 0),  # NOTE: code to be written
            tc=0,
            room=r,
        )
        for r in rooms
    }

    def f(node: Node, ignore):
        for child in node.room.adjacent:
            if (child.name, node.room.name) in ignore or (
                node.room.name,
                child.name,
            ) in ignore:
                continue
            if name2node[child.name] is node.father:
                continue
            node.children.append(name2node[child.name])
            name2node[child.name].father = node
            f(name2node[child.name], ignore)

    roots = []
    for ignore in ignore_pairs:
        name2node = deepcopy(name2node_original)
        r = rooms[root_idx]
        root = name2node[r.name]
        f(root, ignore)
        roots.append(root)
    return roots


@dataclass
class Node:
    p: float
    father: Node | None
    children: list[Node]
    t: tuple[float, float]
    tc: float  # time to clear the room
    # tc: Callable[float, float]  # since with fire and time the thing is different
    room: Any


@dataclass(frozen=True)
class Node_binary:
    p: float
    father: Node_binary | None
    children: tuple[Node_binary | None, Node_binary | None]
    t: tuple[float, float]
    tc: float  # time to clear the room
    # tc: Callable[float, float]  # since with fire and time the thing is different
    name: str


dummy_counter = 0


def _make_dummy(parent: Node_binary | None) -> Node_binary:
    """Create an intermediate node."""
    global dummy_counter
    dummy_counter += 1
    return Node_binary(
        p=1.0,
        father=parent,
        children=(None, None),
        t=(1.0, 1.0),
        tc=1.0,
        name=f"dummy_{dummy_counter}",
    )


def convert(node: Node, parent_bin: Node_binary | None = None) -> Node_binary:
    """Convert a k-ary Node into a binary Node_binary tree."""
    # First convert this node
    node_bin = Node_binary(
        p=node.p,
        father=parent_bin,
        children=(None, None),
        t=node.t,
        tc=node.tc,
        name=node.room.name if hasattr(node.room, "name") else str(node.room),
    )

    # No children → done
    if not node.children:
        return node_bin

    # Convert children into binary nodes first (without linking)
    bin_children = [convert(child, None) for child in node.children]

    # Now build a left-branching binary chain for these children
    # Attach first child to left
    first = bin_children[0]
    object.__setattr__(first, "father", node_bin)

    if len(bin_children) == 1:
        object.__setattr__(node_bin, "children", (first, None))
        return node_bin

    # More than one child → chain on the right branch
    current_parent = node_bin
    right_node = None

    for child in bin_children[1:]:
        # Create a dummy
        dummy = _make_dummy(current_parent)

        # Link dummy to previous parent
        if current_parent is node_bin:
            object.__setattr__(node_bin, "children", (first, dummy))
        else:
            object.__setattr__(
                current_parent, "children", (child_prev, dummy)
            )  # pyright: ignore

        # Place child on dummy.left
        object.__setattr__(child, "father", dummy)
        object.__setattr__(dummy, "children", (child, None))

        # Move to next
        current_parent = dummy
        child_prev = child

    return node_bin


def dp(
    node: Node_binary, t0: float, route: list[Node_binary] = []
) -> tuple[float, float]:
    """
    (time in sub-tree, score obtained)
    """
    # leaf node
    if node.children[0] is None:
        return (node.tc, node.p / t0)
    # two children
    elif node.children[0] is not None and node.children[1] is not None:
        score0 = node.p / t0
        # config 1
        dt11, score11 = dp(node.children[0], t0 + node.t[0])
        dt12, score12 = dp(node.children[1], t0 + node.t[0] * 2 + node.t[1] + dt11)
        # config 2
        dt21, score21 = dp(node.children[1], t0 + node.t[1])
        dt22, score22 = dp(node.children[0], t0 + node.t[1] * 2 + node.t[0] + dt21)
        if (score11 + score12) > (score21 + score22):
            score1, score2 = score11, score12
            dt1, dt2 = dt11, dt12
        else:
            score1, score2 = score21, score22
            dt1, dt2 = dt21, dt22
        return u(
            node.t[0] * 2 + node.t[1] * 2 + dt1 + dt2 + node.tc,
            score1 + score2 + score0,
        )
    # one child
    elif node.children[0] is not None and node.children[1] is None:
        score0 = node.p / t0
        dt1, score1 = dp(node.children[0], t0 + node.t[0])
        return (
            node.t[0] * 2 + dt1 + node.tc,
            score1 + score0,
        )
    else:
        raise RuntimeError("Bad format")
