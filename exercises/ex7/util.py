import numpy as np

color_label_pairs = [
    ((255, 255, 255),     0),
    ((255, 0, 0),         1),
    ((0, 255, 0),         2),
    ((0, 0, 255),         3),
    ((255, 255, 0),       4),
]

num_labels = len(color_label_pairs)

def color_to_label(mask):
    out = np.zeros(mask.shape[0:2], dtype=np.int32)
    mask = mask.astype(np.uint32)
    mask = 256 * 256 * mask[:, :, 0] + 256 * mask[:, :, 1] + mask[:, :, 2]

    for color, label in color_label_pairs:
        color = 256 * 256 * color[0] + 256 * color[1] + color[2]
        # print(mask.size)
        # print(color.size)
        out += (mask == color) * label

    return out


def label_to_colors(mask):
    out = np.zeros(mask.shape + (3,), dtype=np.int64)
    for color, label in color_label_pairs:
        trues = np.stack([(mask == label)] * 3, axis=-1)
        out += np.tile(color, mask.shape + (1,)) * trues

    out = np.ndarray.astype(out, dtype=np.uint8)
    return out
