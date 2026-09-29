import os, glob
import numpy as np, cv2
from common import OUT_DIR, TILE

IMG = f"{OUT_DIR}/dataset/images"
LBL = f"{OUT_DIR}/dataset/labels"
os.makedirs(f"{OUT_DIR}/streak_checks", exist_ok=True)

n = 0
for f in sorted(glob.glob(f"{LBL}/*.txt")):
    lines = [l.split() for l in open(f) if l.strip()]
    if not any(p[0] == "1" for p in lines):
        continue
    name = os.path.basename(f)[:-4]
    raw = cv2.cvtColor(cv2.imread(f"{IMG}/{name}.png", 0), cv2.COLOR_GRAY2BGR)
    ov = raw.copy()
    for p in lines:
        xy = (np.array(p[1:], float).reshape(-1, 2) * TILE).round().astype(np.int32)
        color = (0, 0, 255) if p[0] == "1" else (255, 120, 0)
        cv2.polylines(ov, [xy], True, color, 1)
    cv2.imwrite(f"{OUT_DIR}/streak_checks/{name}.png", np.hstack([raw, ov]))
    print(name)
    n += 1
print("tiles with streaks:", n)