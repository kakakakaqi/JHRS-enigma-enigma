import matplotlib.pyplot as plt
import numpy as np
import fdsreader


def get_room_fields(sim, room, frame=0):
    """
    Extracts the soot and temperature slices cropped to the room bounds for a given frame.

    Args:
        sim: fdsreader.Simulation object
        room: object with attributes x0, x1, y0, y1, z0, z1
        frame: integer frame index

    Returns:
        dict: {"temperature": array, "soot": array}
    """

    def find_slice(quantity_name, spec=None):
        for s in sim.slices:
            slice_obj = s[0]  # first mesh
            # print(slice_obj.quantity)
            if s.quantity.name == quantity_name:
                if spec is None or getattr(slice_obj.quantity, "spec", None) == spec:
                    return slice_obj
        raise ValueError(f"No slice found for quantity {quantity_name} spec={spec}")

    def crop_slice(slice_obj):
        extent = slice_obj.extent  # [x0,x1,y0,y1,z0,z1]
        data = slice_obj.data[frame]  # single mesh, given frame
        orientation = slice_obj.orientation
        if orientation == 1:
            print("NEIN")
            raise RuntimeError("NEIN")
        elif orientation == 2:
            print("NEIN")
            raise RuntimeError("NEIN")
        elif orientation == 3:
            # Z-normal slice: axes are X,Y
            x_vals = np.linspace(extent[1][0] + 1, extent[1][1] - 1, data.shape[0])
            y_vals = np.linspace(extent[2][0] + 1, extent[2][1] - 1, data.shape[1])
            mask_x = (x_vals > (room.rect.x)) & (x_vals < (room.rect.x + room.rect.l))
            mask_y = (y_vals > (room.rect.y)) & (y_vals < (room.rect.y + room.rect.w))
            combined_mask = mask_x[:, np.newaxis] & mask_y[np.newaxis, :]
            cropped = data[np.ix_(mask_x, mask_y)]
        else:
            raise ValueError(f"Unknown orientation {orientation}")
        return cropped

    temp_slice = find_slice("TEMPERATURE")
    soot_slice = find_slice("SOOT MASS FRACTION")

    return {"temperature": crop_slice(temp_slice), "soot": crop_slice(soot_slice)}
