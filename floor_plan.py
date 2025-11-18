from __future__ import annotations
from math import sqrt
from dataclasses import dataclass, field
from typing import (
    Generator,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
    TypedDict,
)
import drawsvg as draw
import analysis_bridge
from Simulation.watchman import compute_watchman_route
import numpy as np

# ------------ globals ------------

rooms: list[Room] = []
doors: list[Door] = []
fires: list[Fire] = []
total_people = 0

# ------------ utilities ------------

_tot = 0
_T_co = TypeVar("_T_co", covariant=True)


# utility function for the automatic incremental naming of room
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

    def dump_xy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.l, self.w)

    def dump_xyz(self) -> tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.l, self.w, self.h)

    def centroid_xy(self) -> tuple[float, float]:
        return (self.x + self.l / 2, self.y + self.w / 2)


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
class Fire:
    """AI Generated"""

    """Fire source definition for FDS export"""

    x: float  # Center location
    y: float
    z: float
    width: float = 1.0  # Fire patch dimensions (m)
    depth: float = 1.0
    name: str = "FIRE1"
    fuel: str = "POLYURETHANE"  # FDS fuel ID
    hrrpua: float = 500.0  # Heat Release Rate Per Unit Area (kW/m²)
    ramp: list[tuple[float, float]] = field(
        default_factory=lambda: [(0, 0), (10, 1), (600, 1)]  # (time_s, fraction)
    )

    def __post_init__(self):
        global fires
        fires.append(self)


class Discomforts(TypedDict):
    visual: float
    thermal: float
    respiratory: float
    combined: float


@dataclass
class Room:
    rect: Rect
    name: str = field(default_factory=new_id)
    people: int = 0
    doors: list[Door] = field(default_factory=list)
    slices: dict[str, list[analysis_bridge.np.ndarray]] = field(default_factory=dict)
    mean_slice: dict[str, float] = field(default_factory=dict)
    discomforts: Discomforts = field(default_factory=dict)  # type: ignore
    slice_dt: float = float("NaN")
    clear_dist = float("NaN")
    vr = float("NaN")

    def __post_init__(self):
        global rooms, total_people
        rooms.append(self)
        total_people += self.people

    @property
    def adjacent(self) -> Generator[Room]:
        for door in self.doors:
            yield door.room_dest

    def analyze_what_was_analyzed(self, k: float):
        """
        k is thermal sensitivity
        """
        eps = 1e-6
        s = self.mean_slice["soot"]
        t = self.mean_slice["temperature"]
        vr = (
            (622 / (self.mean_slice["soot"] * 1000 * 10**6 + eps)) ** (50 / 49)
        ) * 1000
        self.vr = vr
        # visual discomfort
        self.discomforts["visual"] = max(0, min(1, (10 - self.vr) / 10))
        # thermal discomfort
        self.discomforts["thermal"] = max(
            0,
            (1 + 2.7183 ** (-0.778 * k)) / (1 + 2.7183 ** (-k * (t - 22.778)))
            - 2.7183 ** (-0.778 * k),
        )
        # respiratory discomfort
        DTHRESHOLD = 4 * 10**-9  # kg/m^3
        self.discomforts["respiratory"] = max(
            0, min(1, (s - DTHRESHOLD) / (DTHRESHOLD))
        )
        # combined
        self.discomforts["combined"] = 1 - (1 - self.discomforts["visual"]) * (
            1 - self.discomforts["thermal"]
        ) * (1 - self.discomforts["respiratory"])

    def calc_dist(self, fov: float):
        self.clear_dist = compute_watchman_route(
            self, fov_angle=fov, vis_radius=self.vr
        )


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
        x, y, w, h = room.rect.dump_xy()
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
        x, y, w, h = room.rect.dump_xy()
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
        cx, cy = transform(room.rect.centroid_xy())
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
                    x0, y0 = room.rect.centroid_xy()
                    x1, y1 = other.rect.centroid_xy()
                    dsq = (x0 - x1) ** 2 + (y0 - y1) ** 2
                    dinvsq = 1 / (dsq)
                    room.rect.x += dinvsq * FACTOR * (x0 - x1)
                    room.rect.y += dinvsq * FACTOR * (y0 - y1)
                    other.rect.x -= dinvsq * FACTOR * (x0 - x1)
                    other.rect.y -= dinvsq * FACTOR * (y0 - y1)
    if mode == "tight":
        # snapping all the doors
        # TODO: DO IT
        ...


def save_fds(fname: str, duration: float):
    global rooms, doors

    # ===== FIRE-SPECIFIC CONSTANTS =====
    MESH_RESOLUTION = 1
    SIMULATION_DURATION = duration  # 10 minutes
    MAX_CELLS = 200000
    WALL_THICKNESS = 0.15
    WALL_HEIGHT = 3

    # ===== CALCULATE MESH BOUNDS =====
    if not rooms:
        print("No rooms to export!")
        return

    all_x = [room.rect.x for room in rooms] + [
        room.rect.x + room.rect.l for room in rooms
    ]
    all_y = [room.rect.y for room in rooms] + [
        room.rect.y + room.rect.w for room in rooms
    ]

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min = 0.0
    z_max = WALL_HEIGHT

    padding = 1.0
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    # ===== CALCULATE MESH CELLS =====
    dx = x_max - x_min
    dy = y_max - y_min
    dz = z_max - z_min

    ijk_x = max(1, int(dx / MESH_RESOLUTION))
    ijk_y = max(1, int(dy / MESH_RESOLUTION))
    ijk_z = max(1, int(dz / MESH_RESOLUTION))

    total_cells = ijk_x * ijk_y * ijk_z
    print(f"Mesh cells: {ijk_x} x {ijk_y} x {ijk_z} = {total_cells} total cells")

    if total_cells > MAX_CELLS:
        MESH_RESOLUTION = 0.3
        ijk_x = max(1, int(dx / MESH_RESOLUTION))
        ijk_y = max(1, int(dy / MESH_RESOLUTION))
        ijk_z = max(1, int(dz / MESH_RESOLUTION))

    # ===== BUILD FDS FILE CONTENT =====
    fds_content = f"""&HEAD CHID='{fname.replace('.fds', '')}', TITLE='Dynamic Fire Simulation' /
    
&MESH IJK={ijk_x},{ijk_y},{ijk_z}, XB={x_min:.3f},{x_max:.3f},{y_min:.3f},{y_max:.3f},{z_min:.3f},{z_max:.3f} /

&TIME T_END={SIMULATION_DURATION} /

&REAC ID='PROPANE_REAC'
   FUEL='PROPANE'
   SOOT_YIELD=0.05
   CO_YIELD=0.01
/

&MATL ID='CONCRETE'
   CONDUCTIVITY=1.0
   SPECIFIC_HEAT=0.88
   DENSITY=2200.0
/

&SURF ID='INERT'
   MATL_ID(1,1)='CONCRETE'
   THICKNESS(1)=0.15
   COLOR='GRAY'
/

&SURF ID='FLOOR'
   COLOR='TAN'
/

"""

    # ===== ADD FLOOR =====
    fds_content += f"&OBST XB={x_min:.3f},{x_max:.3f},{y_min:.3f},{y_max:.3f},{z_min:.3f},{z_min:.3f}, SURF_ID='FLOOR' / Main Floor\n"

    # ===== ADD WALLS FOR EACH ROOM =====
    for room in rooms:
        x, y, z, l, w, h = room.rect.dump_xyz()
        wall_height = WALL_HEIGHT

        fds_content += f"&OBST XB={x:.3f},{x+l:.3f},{y:.3f},{y+WALL_THICKNESS:.3f},{z:.3f},{z+wall_height:.3f}, SURF_ID='INERT' / {room.name} South Wall\n"
        fds_content += f"&OBST XB={x:.3f},{x+l:.3f},{y+w-WALL_THICKNESS:.3f},{y+w:.3f},{z:.3f},{z+wall_height:.3f}, SURF_ID='INERT' / {room.name} North Wall\n"
        fds_content += f"&OBST XB={x:.3f},{x+WALL_THICKNESS:.3f},{y:.3f},{y+w:.3f},{z:.3f},{z+wall_height:.3f}, SURF_ID='INERT' / {room.name} West Wall\n"
        fds_content += f"&OBST XB={x+l-WALL_THICKNESS:.3f},{x+l:.3f},{y:.3f},{y+w:.3f},{z:.3f},{z+wall_height:.3f}, SURF_ID='INERT' / {room.name} East Wall\n"

    # ===== ADD DOOR OPENINGS =====
    processed_doors = set()
    door_width = 0.9
    door_height = 2.0

    for door in doors:
        door_id = frozenset([door.room_source.name, door.room_dest.name, door.name])
        if door_id in processed_doors:
            continue
        processed_doors.add(door_id)

        source_room = door.room_source
        sx, sy, sz, sl, sw, sh = source_room.rect.dump_xyz()
        door_x = door.x
        door_y = door.y

        dist_south = abs(door_y - sy)
        dist_north = abs(door_y - (sy + sw))
        dist_west = abs(door_x - sx)
        dist_east = abs(door_x - (sx + sl))

        min_dist = min(dist_south, dist_north, dist_west, dist_east)

        if min_dist == dist_south:
            hole_x1 = door_x - door_width / 2
            hole_x2 = door_x + door_width / 2
            hole_y1 = sy
            hole_y2 = sy + WALL_THICKNESS
        elif min_dist == dist_north:
            hole_x1 = door_x - door_width / 2
            hole_x2 = door_x + door_width / 2
            hole_y1 = sy + sw - WALL_THICKNESS
            hole_y2 = sy + sw
        elif min_dist == dist_west:
            hole_x1 = sx
            hole_x2 = sx + WALL_THICKNESS
            hole_y1 = door_y - door_width / 2
            hole_y2 = door_y + door_width / 2
        else:
            hole_x1 = sx + sl - WALL_THICKNESS
            hole_x2 = sx + sl
            hole_y1 = door_y - door_width / 2
            hole_y2 = door_y + door_width / 2

        fds_content += f"&HOLE XB={hole_x1:.3f},{hole_x2:.3f},{hole_y1:.3f},{hole_y2:.3f},0.0,{door_height:.3f} / Door: {door.name}\n"

        # shishan
        source_room = door.room_dest
        sx, sy, sz, sl, sw, sh = source_room.rect.dump_xyz()
        door_x = door.x
        door_y = door.y

        dist_south = abs(door_y - sy)
        dist_north = abs(door_y - (sy + sw))
        dist_west = abs(door_x - sx)
        dist_east = abs(door_x - (sx + sl))

        min_dist = min(dist_south, dist_north, dist_west, dist_east)

        if min_dist == dist_south:
            hole_x1 = door_x - door_width / 2
            hole_x2 = door_x + door_width / 2
            hole_y1 = sy
            hole_y2 = sy + WALL_THICKNESS
        elif min_dist == dist_north:
            hole_x1 = door_x - door_width / 2
            hole_x2 = door_x + door_width / 2
            hole_y1 = sy + sw - WALL_THICKNESS
            hole_y2 = sy + sw
        elif min_dist == dist_west:
            hole_x1 = sx
            hole_x2 = sx + WALL_THICKNESS
            hole_y1 = door_y - door_width / 2
            hole_y2 = door_y + door_width / 2
        else:
            hole_x1 = sx + sl - WALL_THICKNESS
            hole_x2 = sx + sl
            hole_y1 = door_y - door_width / 2
            hole_y2 = door_y + door_width / 2

        fds_content += f"&HOLE XB={hole_x1:.3f},{hole_x2:.3f},{hole_y1:.3f},{hole_y2:.3f},0.0,{door_height:.3f} / Door: {door.name}\n"

    # ===== ADD FIRE SOURCES =====
    #
    # We generate:
    #   - a SURF for each distinct fire (with HRRPUA and optionally RAMP_Q)
    #   - a VENT acting as the fire surface
    #   - a RAMP_Q entry if specified
    #

    def format_ramp(fire_obj):
        # Converts fire_obj.ramp -> string series of &RAMP lines for that fire
        out = ""
        if not getattr(fire_obj, "ramp", None):
            return out
        for t, frac in fire_obj.ramp:
            out += f"&RAMP ID='RAMP_{fire_obj.name}' T={t:.3f} F={frac:.3f} /\n"
        return out

    if fires:
        fds_content += "\n\n! ===== FIRE DEFINITIONS =====\n"

    for fire in fires:
        # Fire SURF with HRRPUA and optional RAMP
        fds_content += (
            f"&SURF ID='{fire.name}'\n"
            f"   HRRPUA={fire.hrrpua:.1f}\n"
            f"   COLOR='RED'\n"
        )

        if getattr(fire, "ramp", None):
            fds_content += f"   RAMP_Q='RAMP_{fire.name}'\n"

        fds_content += "/\n\n"

        # RAMP lines (optional)
        if getattr(fire, "ramp", None):
            fds_content += format_ramp(fire) + "\n"

        # Fire VENT (used as heat source patch)
        half_w = fire.width / 2
        half_d = fire.depth / 2

        xb_x1 = fire.x - half_w
        xb_x2 = fire.x + half_w
        xb_y1 = fire.y - half_d
        xb_y2 = fire.y + half_d

        fds_content += (
            f"&VENT XB={xb_x1:.3f},{xb_x2:.3f},{xb_y1:.3f},{xb_y2:.3f},"
            f"{fire.z:.3f},{fire.z:.3f} "
            f"SURF_ID='{fire.name}' /\n"
            f"! Fire: {fire.name}\n\n"
        )
    # ===== ADD MONITORING =====
    fds_content += """
&DUMP DT_DEVC=5.0, DT_SLCF=10.0, NFRAMES=100 /
"""
    # Add temperature devices
    for i, room in enumerate(rooms):
        cx, cy = room.rect.centroid_xy()
        fds_content += f"&DEVC XYZ={cx:.3f},{cy:.3f},1.2, QUANTITY='TEMPERATURE', ID='TEMP_{i}' /\n"

    # mid-height for slices
    mid_z = WALL_HEIGHT * 3 / 4  # smoke is mainly high

    # Add slice files (mid-height PBZ slices for smoke and temperature)
    fds_content += f"""
!==================== SLICES (mid-height) ====================

&SLCF PBZ={mid_z:.3f}, QUANTITY='MASS FRACTION', SPEC_ID='SOOT' /
&SLCF PBZ={mid_z:.3f}, QUANTITY='TEMPERATURE' /

&ISOF QUANTITY='TEMPERATURE', VALUE=50.0 /
&ISOF QUANTITY='TEMPERATURE', VALUE=100.0 /
"""

    # Add boundary conditions
    fds_content += f"""
&VENT MB='XMIN', SURF_ID='OPEN' /
&VENT MB='XMAX', SURF_ID='OPEN' /
&VENT MB='YMIN', SURF_ID='OPEN' /
&VENT MB='YMAX', SURF_ID='OPEN' /
! &VENT MB='ZMAX', SURF_ID='OPEN' /  ! NEIN!!, this makes smoke vanish through the 'ceiling'

&TAIL /
"""

    # ===== WRITE FILE =====
    try:
        with open(fname, "w") as f:
            f.write(fds_content)
    except Exception as e:
        print(f"Error writing FDS file: {e}")


def analyze_stuff(sim: analysis_bridge.fdsreader.Simulation):
    global rooms
    for room in rooms:
        i = 0
        soot = []
        temperature = []
        while True:
            try:
                f = analysis_bridge.get_room_fields(sim, room, i)
                soot.append(f["soot"])
                temperature.append(f["temperature"])
                i += 1
            except:
                break
        times = sim.slices[0][0].times
        dt = (times[-1] - times[0]) / (times.size - 1)
        room.slices["soot"] = soot
        room.slices["temperature"] = temperature
        room.slice_dt = dt

        a, b = 0.0661, 0.937
        room.mean_slice["soot"] = 0
        for i in range(len(room.slices["soot"])):
            room.mean_slice["soot"] += np.mean(room.slices["soot"][i]) * a * b**i
        room.mean_slice["temperature"] = 0
        for i in range(len(room.slices["temperature"])):
            room.mean_slice["temperature"] += (
                np.mean(room.slices["temperature"][i]) * a * b**i
            )


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

    # NEW: Add fire sources
    # Fire in the kitchen
    kitchen_fire = Fire(
        x=40,
        y=42.5,
        z=0.0,
        width=1.5,
        depth=1.5,
        name="Kitchen_Fire",
        fuel="POLYURETHANE",
        hrrpua=800.0,
        ramp=[(0, 0), (30, 1), (120, 1)],
    )

    # Smaller fire in living room
    living_fire = Fire(
        x=10,
        y=42.5,
        z=0.0,
        width=1.0,
        depth=1.0,
        name="Living_Fire",
        fuel="WOOD",
        hrrpua=400.0,
        ramp=[(10, 0), (60, 1), (300, 1)],
    )

    # format_rooms("scatter")

    # Save the floor plan (scale=20 makes it slightly smaller than default)
    save("sample_floor_plan.png", scale=20)

    # import os
    #
    # os.system("wslview sample_floor_plan.png")

    save_fds("floor_plan.fds")
