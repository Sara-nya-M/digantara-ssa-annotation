import glob, os
import numpy as np
import cv2
from astropy.io import fits
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_closing
RAW_DIR = "data/raw"
OUT_DIR = "output"
TILE = 1024 # tile size
BG_BLOCK = 64 # background estimate block size (px)
SMOOTH_SIGMA = 1.0 # Gaussian blur sigma applied for object detection
LOW_SNR = 10.0
HIGH_SNR = 20.0
CLOSE_RADIUS = 1
MIN_AREA = 6
STREAK_MIN_LEN = 25
STREAK_MIN_AR = 4.0 # streak minimum length-to-width aspect ratio
STRETCH_LO, STRETCH_HI = -2.0, 10.0 # 8-bit conversion range (in sigma)

CLASS_NAMES = ["star", "streak"] # YOLO class mapping (0: star, 1: streak)


def list_fits():
    files = glob.glob(f"{RAW_DIR}/*.fit*") + glob.glob(f"{RAW_DIR}/*.FIT*")
    return sorted(set(files))

def load_fits(path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return hdu.data.astype(np.float32) # float32 prevents underflow/overflow during subtraction
    raise ValueError("No image data in " + path)

def robust_sigma(a, floor=0.1):
    med = np.median(a)
    dev = np.abs(a - med)
    mad = np.median(dev)
    if mad > floor:
        return med, 1.4826 * mad
    for pct, divisor in [(68.27, 1.0), (84.0, 1.405), (95.0, 1.96), (99.0, 2.576)]:
        q = np.percentile(dev, pct)
        if q > floor:
            return med, q / divisor
    return med, 1.0

def preprocess(img):
    """Background subtract + noise normalise. Returns z (1 unit = 1 noise sigma)."""
    H, W = img.shape
    B = BG_BLOCK
    Hp, Wp = -(-H // B) * B, -(-W // B) * B
    padded = np.pad(img, ((0, Hp - H), (0, Wp - W)), mode="reflect")
    blocks = padded.reshape(Hp // B, B, Wp // B, B)
    bg_small = np.median(blocks, axis=(1, 3)).astype(np.float32) # Median filtering ignores point sources (stars)
    bg_small = cv2.GaussianBlur(bg_small, (0, 0), 1.5)
    bg = cv2.resize(bg_small, (Wp, Hp), interpolation=cv2.INTER_CUBIC)[:H, :W]
    sub = img - bg
    _, sigma = robust_sigma(sub[::4, ::4])
    return (sub / max(sigma, 1e-6)).astype(np.float32), bg, sigma

def to8bit(z):
    v = (z - STRETCH_LO) / (STRETCH_HI - STRETCH_LO) * 255.0
    return np.clip(v, 0, 255).astype(np.uint8)

ZERO_GRAY = int(to8bit(np.zeros(1, np.float32))[0]) # padding color value

def detect(z):
    """Returns lab (object id per pixel) and cls (class per id: 0 reject, 1 star, 2 streak)."""
    sm = cv2.GaussianBlur(z, (0, 0), SMOOTH_SIGMA)
    _, s = robust_sigma(sm[::4, ::4], floor=1e-4)
    mask = apply_hysteresis_threshold(sm, LOW_SNR * max(s, 1e-6), HIGH_SNR * max(s, 1e-6))
    mask = binary_closing(mask, disk(CLOSE_RADIUS))
    lab = label(mask, connectivity=2).astype(np.int32)
    cls = np.zeros(lab.max() + 1, np.uint8)

    for r in regionprops(lab):
        if r.area < MIN_AREA:
            continue
        major = r.major_axis_length
        minor = max(r.minor_axis_length, 1.0)
        is_streak = major >= STREAK_MIN_LEN and (major / minor) >= STREAK_MIN_AR
        cls[r.label] = 2 if is_streak else 1

    return lab, cls