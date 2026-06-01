# SyntheticCellExtrusion

Synthetic data generation framework for simulating **cell tissues with extrusion events** and extracting labeled image patches for downstream analysis (e.g., detection, classification, morphology studies).

This repository combines:
- Parameters estimated from **real microscopy data**
- A **synthetic embryo generator**
- A pipeline for creating **synthetic extrusion events** in embryo

---

## Repository Structure

```
SyntheticCellExtrusion/
│
├── real_data/
│   ├── 01/
│   ├── 02/
│   ├── ...
│   ├── 01_GT/
│   ├── 02_GT/
│   └── ...
│
├── main/
│   ├── config/
│   └── GeneratorPatchData.py
│
└── README.md
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

### Package Dependencies

```txt
numpy
scipy
scikit-image
opencv-python
matplotlib
tqdm
tifffile
```

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
```




