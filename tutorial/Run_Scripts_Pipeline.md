# Run Scripts Pipeline

This document explains the execution flow of the two launcher scripts:

- `run.sh` (local interactive pipeline)

---

## 1) `run.sh` Pipeline (Local)

Run command:

```bash
bash run.sh
```

### Step-by-step flow

1. Clear screen and print banner/copyright info.
2. Ask user to select:
   - Backbone (`ResNet50`, `ResNet50-IBN`, `DenseNet121`, `Swin`)
   - Dataset (`Market-1501`, `DukeMTMC-reID`, `MSMT17`, `Custom path`)
   - Loss (`CE`, `Circle`, `Triplet`)
3. Fix GPU as `gpu_ids=0`.
4. Ask `which_epoch` and `run_name`.
5. Generate one shared `run_id` (`python tool/run_id.py`).
6. Clear screen, print a summary, and wait for confirmation (`[Y/n]`).
7. Execute pipeline:
   - `[1/4]` install dependencies: `pip install -r requirements.txt`
   - `[2/4]` dataset stage:
     - check whether prepared data exists (`train/query/gallery`)
     - if missing:
       - Market/Duke: try auto-download from Google Drive + unzip + `prepare.py`
       - MSMT17: require local raw data, then `prepare.py`
       - Custom: require already prepared path
     - if auto-download/preparation fails, stop and print manual preparation guide
   - `[3/4]` training: `python train.py ... --run_id <same_id>`
   - `[4/4]` testing: `python test.py ... --run_id <same_id>`
8. Print output artifact paths.

### Shared run id

`run.sh` passes one same `run_id` into both `train.py` and `test.py`, so logs from the same run can be correlated.

---

## 2) Typical Usage

Clone repo first:

```bash
git clone https://github.com/wilson-lyc/Person_reID
cd Person_reID
```

Run:

```bash
bash run.sh
```

---

## 3) Manual Dataset Preparation (When Auto Download Fails)

If Google Drive is inaccessible, `run.sh` stops and prints manual instructions.  
You can prepare datasets manually with:

```bash
python prepare.py --dataset market --download_path ./Market
python prepare.py --dataset duke --download_path ./DukeMTMC-reID
python prepare.py --dataset msmt17 --download_path ./MSMT17_V1
```

After preparation is complete, rerun `run.sh`.
