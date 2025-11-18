import fdsreader
from floor_plan import *

# ------------ definition ------------

office1 = Room(Rect(0, 0, 0, 8.75, 6.25, 3), "O1", 0)
office2 = Room(Rect(8.75, 0, 0, 8.75, 6.25, 3), "O2", 0)
office3 = Room(Rect(17.5, 0, 0, 8.75, 6.25, 3), "O3", 5)

hallway = Room(Rect(0, 6.25, 0, 26.25, 3.75, 3), "Hallway")

office4 = Room(Rect(0, 10, 0, 8.75, 6.25, 3), "O4", 10)
office5 = Room(Rect(8.75, 10, 0, 8.75, 6.25, 3), "O5", 0)
office6 = Room(Rect(17.5, 10, 0, 8.75, 6.25, 3), "O6", 5)

add_door(hallway, (7.5, 0), office1, (7.5, 6.25))
add_door(hallway, (8.75 * 3 / 2, 0), office2, (8.75 / 2, 6.25))
add_door(hallway, (8.75 * 2 + (8.75 - 7.5), 0), office3, ((8.75 - 7.5), 6.25))

add_door(hallway, (7.5, 3.75), office4, (7.5, 0))
add_door(hallway, (8.75 * 3 / 2, 3.75), office5, (8.75 / 2, 0))
add_door(hallway, (8.75 * 2 + (8.75 - 7.5), 3.75), office6, ((8.75 - 7.5), 0))


Fire(
    8.75 * 3 / 2,
    6.25 / 2 + 6.25,
    0,
    width=1.5,
    depth=1.5,
    fuel="POLYURETHANE",
    hrrpua=800,
    ramp=[(0, 0), (30, 1), (120, 1)],
)

save_fds("standard.fds", 600)
save("standard_floor_map.png", scale=25)

k = 0.1
root = hallway

# ------------ template ------------

compute_dijkstra()

analyze_stuff(fdsreader.Simulation("./standard"), k, root.name)

i_hate_watchman(False)

t, s, r = dp(root.idx)

assert t is not None
assert s is not None
assert r is not None

print(f"time: {t}")
for x in r:
    print(x.name)
