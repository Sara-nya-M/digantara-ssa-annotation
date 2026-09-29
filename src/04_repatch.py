import os, json, glob
import numpy as np, cv2
from common import *

IMG_DIR = f"{OUT_DIR}/dataset/images"
LBL_DIR = f"{OUT_DIR}/dataset/labels"
REP_DIR = f"{OUT_DIR}/repatched_full"

os.makedirs(REP_DIR, exist_ok=True)

COLORS = {0: (255, 120, 0), 1: (0, 0, 255)} # BGR: star blue, streak red

for jf in sorted(glob.glob(f"{OUT_DIR}/dataset/info/*.json")):
    info = json.load(open(jf))

    stem, H, W = info["stem"], info["H"], info["W"]
    rows, cols = info["rows"], info["cols"]

    canvas = np.zeros((rows * TILE, cols * TILE), np.uint8)
    polys = []

    for r in range(rows):
        for c in range(cols):
            name = f"{stem}_r{r:02d}_c{c:02d}"

            tile = cv2.imread(
                f"{IMG_DIR}/{name}.png",
                cv2.IMREAD_GRAYSCALE
            )

            canvas[
                r * TILE:(r + 1) * TILE,
                c * TILE:(c + 1) * TILE
            ] = tile

            for line in open(f"{LBL_DIR}/{name}.txt"):
                p = line.split()

                xy = np.array(
                    p[1:], dtype=float
                ).reshape(-1, 2) * TILE

                xy += [c * TILE, r * TILE]

                polys.append(
                    (int(p[0]), xy.round().astype(np.int32))
                )

    full = canvas[:H, :W] # padding vetti

    assert full.shape == (H, W), "size mismatch!"

    bgr = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)

    layer = bgr.copy()

    for k, xy in polys:
        cv2.fillPoly(layer, [xy], COLORS[k])

    overlay = cv2.addWeighted(layer, 0.6, bgr, 0.4, 0)

    cv2.imwrite(
        f"{REP_DIR}/{stem}_clean.png",
        full
    )

    cv2.imwrite(
        f"{REP_DIR}/{stem}_overlay.jpg",
        overlay,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    print(stem, full.shape, "objects:", len(polys))