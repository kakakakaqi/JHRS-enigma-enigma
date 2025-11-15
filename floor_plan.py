from __future__ import annotations
from math import sqrt
from dataclasses import dataclass, field
from typing import (
    Generator,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
)
import drawsvg as draw

# ------------ globals ------------

rooms: list[Room] = []
doors: list[Door] = []

# ------------ utilities ------------

_tot = 0
_T_co = TypeVar("_T_co", covariant=True)


# utiliy function for the automatic incremental naming of room
def new_id():
    global _tot
    _tot += 1
    return str(_tot)


class Has_pos(Protocol):
    x: float | int
    y: float | int


def dist(a: Has_pos, b: Has_pos) -> float:
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# ------------ the actual stuff ------------


@dataclass
class Rect:
    x: float; y: float; z: float  # fmt: skip
    l: float; w: float; h: float  # fmt: skip

    def dump_planar(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.l, self.w)

    def centroid_planar(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


@dataclass
class Door:
    # the room it leads to
    room_source: Room
    room_dest: Room

    # this is under the assumption doors are only so wide that
    # firefighters may only pass through if aligned with the center
    # NOTE: technically these should be identical since the two rooms are connected
    # but for the simplicity of aligning the rooms just think this as doraemon's door
    x_offset: float; y_offset: float  # fmt: skip
    x_new_offset: float; y_new_offset: float  # fmt: skip

    name: str = ""

    @property
    def x(self) -> float:
        return self.room_source.rect.x + self.x_offset

    @property
    def y(self) -> float:
        return self.room_source.rect.y + self.y_offset

    @property
    def x_new(self) -> float:
        return self.room_dest.rect.x + self.x_new_offset

    @property
    def y_new(self) -> float:
        return self.room_dest.rect.y + self.y_new_offset

    def __post_init__(self):
        global doors
        doors.append(self)


@dataclass
class Room:
    rect: Rect
    name: str = field(default_factory=new_id)
    doors: list[Door] = field(default_factory=list)

    def __post_init__(self):
        global rooms
        rooms.append(self)

    @property
    def adjacent(self) -> Generator[Room]:
        for door in self.doors:
            yield door.room_dest


def add_door(
    room1: Room, p1: Sequence[float], room2: Room, p2: Sequence[float], name: str = ""
):
    d1 = Door(room1, room2, p1[0], p1[1], p2[0], p2[1], name)
    d2 = Door(room2, room1, p2[0], p2[1], p1[0], p1[1], name)
    room1.doors.append(d1)
    room2.doors.append(d2)


def save(
    f_name: str, scale=25, wall_scale=25 / 2, text_scale=25, line_scale=3, padding=25
):
    """
    AI GENERATED
    """
    global rooms, doors

    # 1. Calculate bounds in original coordinate space
    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")

    has_content = bool(rooms or doors)

    for room in rooms:
        x, y, w, h = room.rect.dump_planar()
        min_x, max_x = min(min_x, x), max(max_x, x + w)
        min_y, max_y = min(min_y, y), max(max_y, y + h)

    for door in doors:
        min_x = min(min_x, door.x, door.x_new)
        max_x = max(max_x, door.x, door.x_new)
        min_y = min(min_y, door.y, door.y_new)
        max_y = max(max_y, door.y, door.y_new)

    # Default bounds if empty
    if not has_content:
        min_x, max_x, min_y, max_y = 0.0, 100.0, 0.0, 100.0

    # 2. Calculate canvas dimensions (no origin shifting needed)
    canvas_width = (max_x - min_x + 2 * padding) * scale
    canvas_height = (max_y - min_y + 2 * padding) * scale

    # 3. Create drawing with default origin (0,0)
    drawing = draw.Drawing(width=canvas_width, height=canvas_height)

    # White background covering entire canvas
    drawing.append(draw.Rectangle(0, 0, canvas_width, canvas_height, fill="white"))

    # 4. Transform: translate to padding offset, scale, and FLIP Y-axis
    # SVG has origin at top-left, Y increases downward
    def transform(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        tx = (x - min_x + padding) * scale
        ty = (max_y - y + padding) * scale  # Flip Y
        return (tx, ty)

    # 5. Draw rooms
    for room in rooms:
        x, y, w, h = room.rect.dump_planar()
        tx, ty = transform((x, y))
        tw, th = w * scale, h * scale

        # Room rectangle
        drawing.append(
            draw.Rectangle(
                tx,
                ty - th,  # Adjust Y because SVG rects are positioned by top-left
                tw,
                th,
                stroke_width=wall_scale,
                stroke="black",
                fill="none",
            )
        )

        # Room name (centered)
        cx, cy = transform(room.rect.centroid_planar())
        drawing.append(
            draw.Text(
                room.name,
                text_scale,
                cx,
                cy,
                text_anchor="middle",
                dominant_baseline="middle",
            )
        )

    # 6. Draw doors and labels
    for door in doors:
        # Clear Walls
        # NOTE: probably do something better in the future than this
        x = door.x
        y = door.y
        x, y = transform((x, y))
        drawing.append(
            draw.Rectangle(x - scale, y - scale, scale * 2, scale * 2, fill="white")
        )
        x = door.x_new
        y = door.y_new
        x, y = transform((x, y))
        drawing.append(
            draw.Rectangle(x - scale, y - scale, scale * 2, scale * 2, fill="white")
        )

        # Door line
        x1, y1 = transform((door.x, door.y))
        x2, y2 = transform((door.x_new, door.y_new))
        drawing.append(
            draw.Line(
                x1,
                y1,
                x2,
                y2,
                stroke="red",
                stroke_width=line_scale,
            )
        )

        # Door name (midpoint)
        label_x = (door.x + door.x_new) / 2
        label_y = (door.y + door.y_new) / 2
        lx, ly = transform((label_x, label_y))
        drawing.append(
            draw.Text(
                door.name,
                text_scale,
                lx,
                ly,
                text_anchor="middle",
                dominant_baseline="middle",
                fill="red",
            )
        )

    drawing.save_png(f_name)


def format_rooms(mode: Literal["scatter"] | Literal["tight"]):
    global rooms, doors
    if mode == "scatter":
        # repulsion
        ITERS = 10
        FACTOR = 10
        for _ in range(ITERS):
            for room in rooms:
                for other in rooms:
                    if other is room:
                        continue
                    x0, y0 = room.rect.centroid_planar()
                    x1, y1 = other.rect.centroid_planar()
                    dsq = (x0 - x1) ** 2 + (y0 - y1) ** 2
                    dinvsq = 1 / (dsq)
                    room.rect.x += dinvsq * FACTOR * (x0 - x1)
                    room.rect.y += dinvsq * FACTOR * (y0 - y1)
                    other.rect.x -= dinvsq * FACTOR * (x0 - x1)
                    other.rect.y -= dinvsq * FACTOR * (y0 - y1)
    if mode == "tight":
        # snapping all the doors
        ...


def save_fds(fname: str):
    f = open(fname, "w")
    for room in rooms:
        ...


# ------------ main ------------


if __name__ == "__main__":
    """AI GENERATED"""
    # Create a sample floor plan: a simple house with central hallway

    hallway = Room(Rect(20, 0, 0, 10, 50, 4), "Hallway")
    living_room = Room(Rect(0, 35, 0, 20, 15, 4), "Living Room")
    bedroom = Room(Rect(0, 15, 0, 20, 15, 4), "Bedroom")
    kitchen = Room(Rect(30, 35, 0, 20, 15, 4), "Kitchen")
    bathroom = Room(Rect(30, 15, 0, 15, 10, 4), "Bathroom")
    entryway = Room(Rect(20, -10, 0, 10, 10, 4), "Entryway")

    # Add doors connecting rooms to hallway (on shared walls)
    # Each add_door() creates a bidirectional connection
    add_door(hallway, (0, 42.5), living_room, (20, 7.5), "Hall-Living")  # Left wall
    add_door(hallway, (10, 42.5), kitchen, (0, 7.5), "Hall-Kitchen")  # Right wall
    add_door(hallway, (0, 22.5), bedroom, (20, 7.5), "Hall-Bedroom")  # Left wall
    add_door(hallway, (10, 20), bathroom, (0, 5), "Hall-Bathroom")  # Right wall
    add_door(hallway, (5, 0), entryway, (5, 10), "Hall-Entry")  # Bottom/top wall

    format_rooms("scatter")

    # Save the floor plan (scale=20 makes it slightly smaller than default)
    save("sample_floor_plan.png", scale=20)

    import os

    os.system("wslview sample_floor_plan.png")
