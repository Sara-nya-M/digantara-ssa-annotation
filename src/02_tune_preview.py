import sys, os
import numpy as np, cv2
from common import *

path, r, c = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

img = load_fits(path)
z, bg, sigma = preprocess(img)
lab, cls = detect(z)
from skimage.measure import regionprops
for rp in regionprops(lab):
    if cls[rp.label] == 2:
        cy, cx = rp.centroid
        print("streak -> tile row", int(cy) // TILE, "col", int(cx) // TILE,
              "| length", round(rp.major_axis_length, 1))

print("noise sigma:", round(float(sigma), 2))
print("stars :", int((cls == 1).sum()), " streaks:", int((cls == 2).sum()), "(whole image)")

y0, x0 = r * TILE, c * TILE

g = to8bit(z)[y0:y0 + TILE, x0:x0 + TILE]
k = cls[lab[y0:y0 + TILE, x0:x0 + TILE]]

rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

rgb[k == 1] = (255, 120, 0)  # star = blue
rgb[k == 2] = (0, 0, 255)    # streak = red

side = np.hstack([
    cv2.cvtColor(g, cv2.COLOR_GRAY2BGR),
    rgb
])

os.makedirs(OUT_DIR, exist_ok=True)

name = os.path.splitext(os.path.basename(path))[0]

out = f"{OUT_DIR}/preview_{name}_r{r}_c{c}.png"

cv2.imwrite(out, side)

print("saved", out)