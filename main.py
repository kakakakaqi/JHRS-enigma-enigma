import fdsreader
from floor_plan import *

# ------------ definition ------------

lobby = Room(Rect(20, -10 - 10, -3, 20, 20, 3), "lobby")
elevator = Room(Rect(20, -10, 0, 10, 10, 4), "Elevator")
hallway = Room(Rect(20, 0, 0, 10, 50, 4), "Hallway")
living_room = Room(Rect(0, 35, 0, 20, 15, 4), "Living Room", people=1)
bedroom = Room(Rect(0, 15, 0, 20, 15, 4), "Bedroom", people=2)
kitchen = Room(Rect(30, 35, 0, 20, 15, 4), "Kitchen")
bathroom = Room(Rect(30, 15, 0, 15, 10, 4), "Bathroom")

add_door(lobby, (5, 20), elevator, (5, 10), "ele")
add_door(hallway, (0, 42.5), living_room, (20, 7.5), "Hall-Living")
add_door(hallway, (10, 42.5), kitchen, (0, 7.5), "Hall-Kitchen")
add_door(hallway, (0, 22.5), bedroom, (20, 7.5), "Hall-Bedroom")
add_door(hallway, (10, 20), bathroom, (0, 5), "Hall-Bathroom")
add_door(hallway, (5, 0), elevator, (5, 10), "Hall-Entry")


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

save_fds("V3.fds", 600)
save("V3.png", scale=25, padding=5)

k = 0.7
root = elevator

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
