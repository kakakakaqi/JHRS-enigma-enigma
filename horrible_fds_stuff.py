import matplotlib.pyplot as plt
import matplotlib.animation as animation
import fdsreader
import pandas as pd

sim = fdsreader.Simulation("./V3")

for slice in sim.slices:
    print(
        f"Slice Type [2D/3D]: {slice.type}\n  Quantity: {slice.quantity.name}\n",
        f" Physical Extent: {slice.extent}\n  Orientation [1/2/3]: {slice.orientation}\n",
    )

data = sim.slices[0][0].data

df = pd.DataFrame(data[60])
df.to_csv("slice.csv")
