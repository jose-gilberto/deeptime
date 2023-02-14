from __future__ import annotations

import glob

from PIL import Image

frames = []
imgs = glob.glob('./representation/logs/*.png')

for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save(
    'linear_repr.gif',
    format='GIF',
    append_images=frames[1:],
    save_all=True,
    duration=500,
    loop=0
)
