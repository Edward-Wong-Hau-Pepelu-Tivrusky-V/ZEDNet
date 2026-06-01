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

Contains experimentally acquired embryo image volumes and corresponding segmentation masks.

```text
real_data/
├── 01/
├── 01_GT/
├── 03/
└── 03_GT/
```

#### Raw image data

Folders (`01`, `03`, etc.) contain microscopy image stacks used for analysis.

Example:

```text
01/
├── 0939-0415-0160.tif
├── 1360-0340-0532.tif
└── 1590-0315-0307.tif
```

#### Ground truth segmentations

Ground truth segmentation masks generated using Cellpose.

```text
01_GT/SEG/
├── cellpose_seg_0939-0415-0160.tif
├── cellpose_seg_1360-0340-0532.tif
├── cellpose_seg_1590-0315-0307.tif
└── RoiSet.zip
```

`RoiSet.zip` contains manually curated ROIs used for validation and synthetic data generation.

---

## Synthetic Data Generation

The synthetic data generation pipeline creates realistic extrusion and control examples for CNN training.

```text
Sythetic_data_generation/
├── config/
├── data_generator/
├── scripts/
├── train_data/
└── utils/
```

### Configuration

```text
config/
├── global_parameters.yaml
└── synth_parameters.yaml
```

Contains all parameters controlling:

* Tissue sampling
* Extrusion simulation
* Intensity perturbations
* Dataset generation

---

### Data Generator

```text
data_generator/
```

Main scripts used for synthetic patch generation.

#### Core files

| File                                 | Description                                       |
| ------------------------------------ | ------------------------------------------------- |
| `synthetic_generator.py`             | Main synthetic data generation pipeline           |
| `GeneratorPatchData_more_homogen.py` | Generates extrusion patches from sampled cells    |
| `GeneratorSampler.py`                | Samples cells from segmented embryos              |
| `GeneratorTissueData.py`             | Creates tissue datasets used for patch generation |

#### Sampled Data

```text
sampled_data/data_WILL/
```

Contains sampled cell geometries extracted from real embryo segmentations and stored as pickle files.

Example:

```text
cellpose_seg_0939-0415-0160.pkl
```

---

### Generated Training Data

```text
train_data/
├── control/
└── extrusion/
```

Contains synthetic training patches.

#### Control patches

```text
control/
├── 001.tif
├── 002.tif
└── ...
```

#### Extrusion patches

```text
extrusion/
├── 001.tif
├── 002.tif
└── ...
```

These datasets are used directly for CNN training.

---

### Utility Functions

```text
utils/
├── load_config.py
└── setup_logger.py
```

Provides configuration loading and logging functionality used throughout the pipeline.

---

### Generation Scripts

```text
scripts/
├── GenerateData.sh
└── Sampling.sh
```

Shell scripts for reproducing the sampling and synthetic data generation pipeline.

Example:

```bash
bash scripts/Sampling.sh
bash scripts/GenerateData.sh
```

---

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
