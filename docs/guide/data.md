# Data pipeline

Code: [`src/trimodal/data.py`](../../src/trimodal/data.py) (ported from notebook 04).

## TR-6 in one paragraph

Six produce types (banana, carrot, guava, Indian gooseberry, mango, tomato), one not-spoiled and one
spoiled specimen of each, photographed with an sRGB camera and a thermal camera and measured with
methane sensors up to three times a day (morning, afternoon, evening). After grouping sessions into
days there are **112 day entries** (76 not spoiled, 36 spoiled) in **12 trajectories**. Some sessions
are missing one or more sensors, which is what the model is built to handle.

| Produce | Days | Not spoiled | Spoiled | Train | Val | Test |
|---|---|---|---|---|---|---|
| Banana | 7 | 3 | 4 | 3 | 2 | 2 |
| Carrot | 9 | 2 | 7 | 6 | 2 | 1 |
| Guava | 10 | 4 | 6 | 6 | 2 | 2 |
| Indian gooseberry | 30 | 18 | 12 | 21 | 5 | 4 |
| Mango | 8 | 5 | 3 | 4 | 2 | 2 |
| Tomato | 48 | 44 | 4 | 33 | 8 | 7 |
| **Total** | **112** | **76** | **36** | **73** | **21** | **18** |

## The manifest

`Code/Manifest/manifests/master_manifest.csv` has one row per (produce, label, date). Notebook 01
builds it; `splits_info.json` records the split (seed 42, 70 / 15 / 15, stratified by produce and
label). Columns used by the code:

| Column(s) | Meaning |
|---|---|
| `fruit`, `label` | Produce type; `not_spoiled` or `spoiled` |
| `actual_date` | `YYYYMMDD`, matched against image file names |
| `corrected_day_index` | Day number within the trajectory (orders the sequence) |
| `srgb_<session>_count`, `ir_<session>_count` | Images captured in that session (0 = missing) |
| `methane_<session>_present` | `TRUE` if the gas reading exists |
| `methane_<session>_ppm`, `_std`, `_left`, `_right` | Mean, spread and the two sensors' readings |
| `days_until_spoilage` | Regression target |
| `exclude` | Rows marked `true` are dropped |
| `split` | `train`, `val` or `test` |

The split is by **day**, so every specimen appears in all three splits. The main test therefore
measures generalisation to later or earlier days of partly seen specimens; generalisation to an
unseen produce type is tested separately (notebook 09).

## From rows to batches

1. **Image index.** File names in `Classified/<Fruit>/<Not_spoiled|Spoiled>/sRGB_images/`
   (`YYYYMMDD_HHMMSS*.jpg`) and `IR_fusion_images/` (`YYYYMMDD_HHMMSS_<tmin>C_<tmax>C.jpg`) are
   bucketed into sessions by hour: morning 05–11, afternoon 12–16, evening 16–19. The spoiled
   banana's thermal images are read from `Normal/Banana/` where TR-6 stores them.
2. **Sessions.** For each manifest row and session, record whether sRGB, thermal and gas exist. The
   gas vector is `[mean ppm, std, left, right, |left − right|, session/2]`.
3. **Days.** Sessions of the same (produce, label, day) are merged: the first available sRGB image,
   the first available thermal image, and the mean of the available gas vectors.
4. **Trajectories.** Days of the same (produce, label) *within one split* are ordered. For each day
   the gas encoder also gets `delta`, the number of days since the last gas reading. A trajectory's
   label and regression target are those of its last day.
5. **Batches.** Trajectories are padded to the longest in the batch. Each batch carries the image
   tensors, the availability masks `rgb_avail`, `ir_avail`, `gas_mask`, the gas gaps `gas_delta`,
   and `pad_mask` (True on padding).

Split sizes: train 12 trajectories / 73 days, validation 12 / 21, test 11 / 18 (carrot has no
not-spoiled test day).

## Training-time extras

- **Augmentation:** one offline copy of every training trajectory, horizontally flipped and rotated
  by an angle drawn from [−15°, 15°]. The same angle is used for every day of a copy.
- **Class balance:** a weighted sampler draws spoiled and not-spoiled trajectories equally often, on
  top of the class weights (1.0, 2.3) in the loss.
- **Calibration pool:** the train and validation trajectory lists side by side (24 trajectories),
  used only to choose the final decision threshold.
