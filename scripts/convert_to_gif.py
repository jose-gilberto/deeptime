from __future__ import annotations

import glob

from PIL import Image

frames = []
imgs = glob.glob('../docs/notebooks/oneclass/logs/conv/*.png')

for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save(
    'conv_occ_explu.gif',
    format='GIF',
    append_images=frames[1:],
    save_all=True,
    duration=1000,
    loop=0
)
