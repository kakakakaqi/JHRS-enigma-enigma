import fdsreader
from floor_plan import *

# --- Rooms --------------------------------------------------------------

rooms = {}


def add_room(n):
    idx = n - 1
    x = 5 * (idx % 4)
    y = 5 * (idx // 4)
    rooms[n] = Room(Rect(x, y, 0, 5, 5, 3), f"R{n}")


for i in range(1, 25):
    add_room(i)


# --- Door helper --------------------------------------------------------


def door_between(r1, side1, r2, side2):
    # Side coordinates are wall-centered:
    # N = (2.5, 5), S = (2.5, 0), E = (5, 2.5), W = (0, 2.5)
    def pos(side):
        if side == "N":
            return (2.5, 5.0)
        if side == "S":
            return (2.5, 0.0)
        if side == "E":
            return (5.0, 2.5)
        if side == "W":
            return (0.0, 2.5)
        raise RuntimeError("bozo")

    add_door(rooms[r1], pos(side1), rooms[r2], pos(side2))


# --- Doors --------------------------------------------------------------
# R1: N
door_between(1, "N", 5, "S")

# R2: N, E
door_between(2, "N", 6, "S")
door_between(2, "E", 3, "W")

# R3: N, W
door_between(3, "N", 7, "S")
door_between(3, "W", 2, "E")

# R4: N
door_between(4, "N", 8, "S")

# R5: N, E, S
door_between(5, "N", 9, "S")
door_between(5, "E", 6, "W")
door_between(5, "S", 1, "N")

# R6: S, E, W
door_between(6, "S", 2, "N")
door_between(6, "E", 7, "W")
door_between(6, "W", 5, "E")

# R7: S, W
door_between(7, "S", 3, "N")
door_between(7, "W", 6, "E")

# R8: N, S
door_between(8, "N", 12, "S")
door_between(8, "S", 4, "N")

# R9: N, E, S
door_between(9, "N", 13, "S")
door_between(9, "E", 10, "W")
door_between(9, "S", 5, "N")

# R10: E, W
door_between(10, "E", 11, "W")
door_between(10, "W", 9, "E")

# R11: N, E, W
door_between(11, "N", 15, "S")
door_between(11, "E", 12, "W")
door_between(11, "W", 10, "E")

# R12: S, W
door_between(12, "S", 8, "N")
door_between(12, "W", 11, "E")

# R13: E, S
door_between(13, "E", 14, "W")
door_between(13, "S", 9, "N")
door_between(13, "N", 17, "S")

# R14: E, W
door_between(14, "E", 15, "W")
door_between(14, "W", 13, "E")

# R15: E, W, S
door_between(15, "E", 16, "W")
door_between(15, "W", 14, "E")
door_between(15, "S", 11, "N")

# R16: W
door_between(16, "W", 15, "E")

door_between(17, "N", 21, "S")
door_between(21, "E", 22, "W")
door_between(22, "S", 18, "N")
door_between(18, "E", 19, "W")
door_between(19, "E", 20, "W")
door_between(19, "N", 23, "S")
door_between(23, "E", 24, "W")


def add_fire(idx):
    x = 5 * (idx % 4)
    y = 5 * (idx // 4)
    Fire(
        x + 2.5,
        y + 2.5,
        0,
        width=1.5,
        depth=1.5,
        fuel="PROPANE",
        hrrpua=800,
        ramp=[(0, 0), (30, 1), (120, 1)],
        name=str(idx),
    )


add_fire(5)
add_fire(11)
add_fire(14)


save_fds("huh.fds", 600)
save("huh.png", scale=25, padding=5)

k = 0.3
root = rooms[1]

# ------------ template ------------

compute_dijkstra()

analyze_stuff(fdsreader.Simulation("./huh"), k, root.name)

i_hate_watchman(True)
print("watchman done")

t, s, r = dp(root.idx)

assert t is not None
assert s is not None
assert r is not None

print(f"time: {t}")
for x in r:
    print(x.name)
