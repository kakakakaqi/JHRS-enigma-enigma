from __future__ import annotations
from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class Node:
    p: float
    father: Node | None
    children: tuple[Node | None, Node | None]
    t: tuple[float, float]
    tc: float  # time to clear the room
    # tc: Callable[float, float]  # since with fire and time the thing is different
    name: str


def dp(node: Node, t0: float, route: list[Node] = []) -> tuple[float, float]:
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
        return (
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
