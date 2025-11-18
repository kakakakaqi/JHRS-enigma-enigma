import seaborn as sns
import matplotlib.pyplot as plt
import fdsreader
import numpy as np
from floor_plan import *

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

compute_dijkstra()

# format_rooms("scatter")

# Save the floor plan (scale=20 makes it slightly smaller than default)
# save("scattered_floor_plan.png", scale=20)

# save_fds("V4.fds")
#
analyze_stuff(fdsreader.Simulation("./V3"))

for room in rooms:
    room.analyze_what_was_analyzed(0.3)
    room.calc_p_value("Entryway")
    print(f"name: {room.name}")
    print(room.discomforts)
    print(room.p)
    print()
