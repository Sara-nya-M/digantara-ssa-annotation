import sys
import numpy as np
from common import *

path, r, c = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
img = load_fits(path)
z, bg, sigma = preprocess(img)
t = z[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]
print("global noise sigma (ADU):", round(float(sigma), 2))
print("this tile median z     :", round(float(np.median(t)), 2))
print("this tile noise z      :", round(float(robust_sigma(t)[1]), 2))
print("pixels above z=10 (%)  :", round(float((t > 10).mean() * 100), 2))