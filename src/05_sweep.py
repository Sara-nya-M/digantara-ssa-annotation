import sys
import numpy as np, cv2
from common import *
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label

img = load_fits(sys.argv[1])
z, _, _ = preprocess(img)
sm = cv2.GaussianBlur(z, (0, 0), SMOOTH_SIGMA)
_, s = robust_sigma(sm[::4, ::4])
print("smoothed noise sigma:", round(float(s), 3))

for high in [6, 8, 10, 12, 14, 16, 20, 25]:
    low = high * 0.5
    m = apply_hysteresis_threshold(sm, low * s, high * s)
    lab = label(m, connectivity=2)
    areas = np.bincount(lab.ravel())[1:]
    print(f"HIGH={high:>3} LOW={low:>5}: objects(area>=6) = {int((areas >= 6).sum())}")