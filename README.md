# Zebrafish Extrusion Detection with Synthetic Data Generation

This repository contains the code, data generation pipeline, and trained models used for automated detection of cell extrusion events in zebrafish embryos. The workflow combines synthetic training data generation with a 3D convolutional neural network (ZEDNet) to identify extrusion events from volumetric microscopy data.

---

## Repository Structure

```text
.
├── real_data/
├── Sythetic_data_generation/
└── ZEDNet/
```

### `real_data/`
Contains real image sequences and corresponding ground-truth labels.

| Folder | Description |
|--------|-------------|
| `01, 02, ...` | Raw real image sequences |
| `01_GT, 02_GT, ...` | Ground-truth labels associated with each sequence |

These data are used to:
- Estimate biological and imaging parameters
- Guide scaling of synthetic generation parameters

---

## Installation

All code is written in **Python**.

---

## Synthetic Image Generation

Synthetic tissue images are generated using parameters defined in:

```
main/config/
```

Two parameter groups are used:

| Group | Purpose |
|------|---------|
| **global** | Parameters derived from real data |
| **synth** | Parameters controlling synthetic image generation |

---

## SYNTH Parameters (Synthetic Image Generation)

### Shape Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `IMAGE_SIZE` | list | Synthetic image size in pixels |
| `CELL_R *` | list | Range of cell radii (µm) |
| `Z_SCALE *` | float | Scale factor in z-direction (<1 = flatter, >1 = elongated shapes) |
| `CELL_SEPARATION *` | int | Minimum distance between sampled cells |

---

### Sampling Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `DISTMAP_BLUR` | bool | If `True`, blur the distance map |
| `DISTMAP_SIG` | float | Sigma for distance map blur |
| `GAUSSIAN_BLUR` | bool | If `True`, blur final image |
| `GAUSSIAN_SIG` | float or list | Sigma for final image blur |

---

**Notes**

- `*` → Values should be based on real data
---

## Main Pipeline

Core script:

```
main/GeneratorPatchData.py
```

### Processing Steps

1. Generate a new packed cell tissue  
2. Along the center slice of each layer, measure unique integers to identify candidate extrusions  
3. Accept candidate extrusions if:
   - Edge criteria satisfied  
   - Rosette size criteria satisfied  
4. Add extrusions with designated size relative to mean cell radius  
5. Validate extrusion count  
   - If incorrect → restart generation  
6. Rotate and extract patches (extrusion + control)  
7. Save samples and update dataset counter  

---

## Non-Config Parameters (Defined in Code)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_max` | 800 | Maximum cells per layer (high value improves packing) |
| `w_s` | 5 | Sliding window size for unique integer count in 2D slice |
| `w_z` | 1 | Width of layer sampling space |
| `rosette_size_parameter` | 0.7 | Mean radii of rosette cells must exceed this proportion of all cell radii |
| `extrusion_p` | 0.7 | Extrusion size as fraction of mean cell radius |
| `rotation_p` | 0.2 | Fraction of patches rotated between −30° and 30° |

---

## Output

The generator produces:
- Extrusion patches  
- Control patches  

```
train_data/
├── control/
├── extrusion/
```

---

## Running the Generator

Example:

```bash
# Example Training Dataset
python data_generator/GeneratorPatchData.py \
    --N 8 \
    --sampler-dir data_generator/sampled_data/data_WILL/ \
    --output-dir train_data/ \
    --logger log_train_data.txt \
## ZEDNet

Contains the neural network architecture, training scripts, testing scripts, and trained weights.

```text
ZEDNet/
├── Training.py
├── Testing.py
└── ZEDNet.pt
```

### Training

```bash
python Training.py
```

Trains the extrusion classifier using synthetic control and extrusion patches.

### Testing

```bash
python Testing.py
```

Runs sliding-window inference on embryo image volumes and generates probability maps for predicted extrusion events.

### Pre-trained Model

```text
ZEDNet.pt
```

Pre-trained network weights used in the manuscript.

---

## Workflow

### 1. Generate Cell Samples

Extract cell geometries from segmented embryos:

```bash
bash scripts/Sampling.sh
```

### 2. Generate Synthetic Dataset

Create extrusion and control examples:

```bash
bash scripts/GenerateData.sh
```

### 3. Train Network

```bash
python ZEDNet/Training.py
```

### 4. Run Inference

```bash
python ZEDNet/Testing.py
```

---

## Data Format

### Training Patches

Input patches are volumetric TIFF stacks:

```text
(Z, X, Y)
```

Typical patch dimensions:

```text
16 × 44 × 44 voxels
```

### Probability Maps

Inference outputs a lower-resolution probability volume representing the likelihood of extrusion events at each sampled spatial location.

---

## Requirements

Recommended environment:

```text
Python >= 3.9
albumentations
matplotlib
numpy
pandas
scikit-learn
tensorboard
tifffile
torch
tqdm
```
