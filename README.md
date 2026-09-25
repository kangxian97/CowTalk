# CowTalk

CowTalk edits an erroneous Circle of Willis segmentation from a text instruction. The network is a point transformer: an input point cloud (coordinates plus the current label) is compressed to a latent set, fused with the instruction text, and used to label a query point cloud. An optional 3D CNN reads a 128³ volume. That volume is either the multi-class error shape alone, or that shape plus the scan (`--use-image`).

The shapes and instructions are published at [CowTalk on Kaggle](https://www.kaggle.com/datasets/daxianzzz/cowtalk). This repository does not include them. Training loads **point clouds** produced from those shapes by `preprocess.py`.

## Setup

```bash
pip install -r requirements.txt
```

Text fusion uses the multimodal BERT in `albef/` (Hugging Face BERT with the cross-attention changes from [ALBEF](https://github.com/salesforce/ALBEF)). That code is included in this repository. `albef/LICENSE.txt` is the Salesforce license; the BERT file header is Apache-2.0. You still need the `transformers` package so the text encoder can load `bert-base-uncased` weights.

## Data layout

After preprocessing:

```text
data/
  pointclouds/                 # one NPZ per erroneous shape
    topcow_ct_001_0001.npz
  instructions_generated/      # one JSON per shape, keys "1".."13" with "short" and "long"
    topcow_ct_001_0001.json
  splits/
    fold_1.json                # {"subjects": {"train": [...], "val": [...], "test": [...]}}
```

Subject ids are the first three underscore fields (`topcow_ct_001`). Instance ids (`0001`) stay in the file name.

## Shape to point cloud

The model does not read meshes or NIfTI volumes. `preprocess.py` turns each ground-truth / error label pair into the point cloud that training loads.

```bash
python preprocess.py \
    --error-dir /path/to/error_labels \
    --gt-dir /path/to/gt_labels \
    --output-dir data/pointclouds
```

An error file named `topcow_ct_001_0001.nii.gz` is paired with `topcow_ct_001.nii.gz`.

Each NPZ (`cowtalk-pointcloud-v1`) contains:

| key | meaning |
| --- | --- |
| `fg_coords`, `fg_gt_labels`, `fg_error_labels` | every cropped voxel where either label is non-zero |
| `halo_coords` | near-surface shell outside that foreground |
| `volume_shape`, `bbox` | crop size and where it sat in the original volume |
| `modified_resized` | 128³ nearest-neighbor copy of the cropped error labels, for the CNN |

The crop, resize, and halo were recovered by matching the last training cache back to the NIfTI labels. An earlier export (`*_surface_points.npz`, with surface samples and signed distances) is **not** what this model uses; the loader skips those files.

Recovered steps, implemented in `pointcloud.py`:

1. **Crop** to the error-label foreground, padded by `round(0.05 * full_volume_shape)` voxels per axis. The saved end index is exclusive and equals `fg_max + margin` (clipped to the volume). Ground truth is cropped with the same box. Label 15 on the ground truth is rewritten to 13.
2. **Resize** the cropped error labels to 128³ with nearest neighbors (`scipy.ndimage.zoom`, order 0).
3. **Halo**: union of the two cropped labels, dilated twice with 6-connectivity, then the original foreground removed. Query sampling draws one quarter of its foreground quota from this shell and the rest from the union. (An older comment described a half/half split; the code that trained used `fg_count // 4`.)

Training still subsamples this point cloud on the fly (default 10 000 input points at 70% foreground / 30% background, 8 000 query points at 90% / 10%) and randomly drops some instructed error classes. With probability 0.05 the input is the fully corrected shape and the text is a “no error” sentence. Instructions use the **short** wording: the previous coin-flip that was meant to mix in the long wording never fired (`random.uniform(0, 1) > 1`).

## Train

```bash
python main.py --data-root data --fold 1 --input-points 10000 --query-points 8000 --batch-size 2
```

Shape only is the default. To also feed the 128³ scan (files in `data/preprocessed_volume_images/`, same names as the point clouds, array `medical_image`):

```bash
python main.py --data-root data --fold 1 --use-image --input-points 10000 --query-points 8000 --batch-size 2
```

Image-mode checkpoints are named with `_img`, so they do not replace a shape-only run. Inference turns the image branch on when the checkpoint name contains `_img`.

Multi-GPU (edit the Slurm account and partition in `run_multi_gpu.sh` for your cluster):

```bash
COW_DATA_ROOT=/path/to/data sbatch run_multi_gpu.sh
```

Checkpoints are written to `cross_attention_checkpoints/`. Pass `--no-cnn` to drop the volume encoder. Training requires a GPU unless you pass `--allow-cpu`.

## Inference

```bash
COW_DATA_ROOT=/path/to/data COWTALK_CHECKPOINT=/path/to/checkpoint.pth bash inference.sh
```

`inference_volume.py` scores a full crop. `inference_iterative.py` applies the edit repeatedly. Both read the same point-cloud directory.

## Layout

| file | role |
| --- | --- |
| `preprocess.py`, `pointcloud.py` | label volumes → point cloud |
| `dataset.py` | load point clouds, sample training batches |
| `albef/` | multimodal BERT that fuses shape latents with the instruction |
| `models.py`, `conv/` | point transformer, text fusion, optional 3D CNN |
| `engine.py`, `main.py` | training loop |
| `inference_volume.py`, `inference_iterative.py` | evaluation |
| `ARCHITECTURE.md` | layer-by-layer notes for the current network |
