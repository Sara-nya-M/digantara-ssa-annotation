import os, json, csv
import numpy as np, cv2
from common import *

IMG_DIR = f"{OUT_DIR}/dataset/images"
LBL_DIR = f"{OUT_DIR}/dataset/labels"
INFO_DIR = f"{OUT_DIR}/dataset/info"

for d in (IMG_DIR, LBL_DIR, INFO_DIR):
    os.makedirs(d, exist_ok=True)

def pad_to_tiles(a, fill):
    H, W = a.shape
    Hp, Wp = -(-H // TILE) * TILE, -(-W // TILE) * TILE
    return np.pad(a, ((0, Hp - H), (0, Wp - W)),
                  mode="constant", constant_values=fill)

def polygon(m):
    """m = 0/1 mask of ONE object inside a tile -> list of (x, y) points."""
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if len(c) < 3 or cv2.contourArea(c) < 1: # tiny / thin object -> box polygon
        x, y, w, h = cv2.boundingRect(c)
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return [tuple(p) for p in c[:, 0, :]]

summary = []

for path in list_fits():
    stem = os.path.splitext(os.path.basename(path))[0]
    print("Processing", stem)

    img = load_fits(path)
    H, W = img.shape

    z, bg, sigma = preprocess(img)
    lab, cls = detect(z)

    g_pad = pad_to_tiles(to8bit(z), ZERO_GRAY)
    lab_pad = pad_to_tiles(lab, 0)

    rows, cols = g_pad.shape[0] // TILE, g_pad.shape[1] // TILE

    n_star = n_streak = 0

    for r in range(rows):
        for c in range(cols):
            name = f"{stem}_r{r:02d}_c{c:02d}"
            y0, x0 = r * TILE, c * TILE

            cv2.imwrite(
                f"{IMG_DIR}/{name}.png",
                g_pad[y0:y0 + TILE, x0:x0 + TILE]
            )

            tl = lab_pad[y0:y0 + TILE, x0:x0 + TILE]
            lines = []

            for i in np.unique(tl):
                if i == 0 or cls[i] == 0:
                    continue

                pts = polygon((tl == i).astype(np.uint8))
                if pts is None:
                    continue

                k = int(cls[i]) - 1 # 0 = star, 1 = streak

                coords = " ".join(
                    f"{px / TILE:.6f} {py / TILE:.6f}"
                    for px, py in pts
                )

                lines.append(f"{k} {coords}")

            with open(f"{LBL_DIR}/{name}.txt", "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

    n_star, n_streak = int((cls == 1).sum()), int((cls == 2).sum())

    json.dump(
        {"stem": stem, "H": H, "W": W, "rows": rows, "cols": cols},
        open(f"{INFO_DIR}/{stem}.json", "w")
    )

    summary.append([
        stem, H, W, rows * cols, n_star, n_streak
    ])

    print(
        f" {rows}x{cols} tiles | "
        f"stars={n_star} streaks={n_streak}"
    )

with open(f"{OUT_DIR}/summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "H", "W", "tiles", "stars", "streaks"])
    w.writerows(summary)

with open(f"{OUT_DIR}/dataset/data.yaml", "w") as f:
    f.write("path: .\ntrain: images\nval: images\nnames:\n")
    for i, n in enumerate(CLASS_NAMES):
        f.write(f" {i}: {n}\n")

print("DONE")