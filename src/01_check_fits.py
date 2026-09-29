import numpy as np
from common import list_fits, load_fits, robust_sigma
files = list_fits()

print(f"Found {len(files)} files\n")
for f in files:
    img = load_fits(f)
    med, sig = robust_sigma(img[::4, ::4])
    print(f)
    print(f" shape={img.shape} min={img.min():.0f} max={img.max():.0f}")
    print(f" median(background)={med:.1f} noise sigma={sig:.2f}")
    print(f" saturated pixels (>=65535): {(img >= 65535).sum()}")