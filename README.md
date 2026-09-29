# SSA Star / Streak Annotation Pipeline

Pre-process -> tile (1024x1024) -> detect -> YOLO-seg labels -> repatch.

## Run

pip install -r requirements.txt

put FITS files in data/raw/

python src/01_check_fits.py

python src/02_tune_preview.py data/raw/<file>.fits 3 4

python src/03_tile_and_label.py

python src/04_repatch.py

## Classes

0 = star (blob), 1 = streak

## Key parameters (src/common.py)

LOW_SNR, HIGH_SNR, MIN_AREA, STREAK_MIN_LEN, STREAK_MIN_AR