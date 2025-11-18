import fdsreader

sim = fdsreader.Simulation("./V3")

# Pick the first slice to inspect
slc = sim.slices[0]

print("=== RAW EXTENT OBJECT ===")
print(slc.extent)
print()

print("=== DIR(extent) ===")
print(dir(slc.extent))
print()

print("=== extent.__dict__ (if any) ===")
if hasattr(slc.extent, "__dict__"):
    print(slc.extent.__dict__)
else:
    print("extent has no __dict__")
print()

# Try to print all numeric-looking attributes
print("=== Numeric attributes ===")
for name in dir(slc.extent):
    if name.startswith("_"):
        continue
    val = getattr(slc.extent, name)
    if isinstance(val, (float, int)):
        print(f"{name} = {val}")
