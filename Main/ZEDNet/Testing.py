#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% packages 
import matplotlib.pyplot as plt, numpy as np
import tifffile as tiff
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import albumentations as A
p= 0.5
crop_val= 10
# pytorch
import torch 
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix


#%% set device 
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

#%% import ZEDNet

class model(nn.Module):

    def __init__(self):
        super().__init__()

        # Conv layers
        self.conv1 = nn.Conv3d(
            1, 16,
            kernel_size=(1,3,3),
            padding=(0,1,1)
        )

        self.conv2 = nn.Conv3d(
            16, 32,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv3d(
            32, 64,
            kernel_size=3,
            padding=1
        )

        self.conv4 = nn.Conv3d(
            64, 128,
            kernel_size=3,
            padding=1
        )

        # batch norms
        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(128)

        # classifer
        self.fc = nn.Linear(128, 1)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, kernel_size=(1,2,2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, kernel_size=(2,2,2))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool3d(x, kernel_size=(2,2,2))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_avg_pool3d(x, 1)
        x = x.flatten(1)
        x = self.fc(x)

        return x
    
zebnet = model()

zebnet.load_state_dict(
    torch.load("./ZEDNet", map_location=device)
)

zebnet.to(device)
zebnet.eval()

#%%%% testing model on synthetic data using synetheticpatch  test set. 

# Defining data class for storage
# data storage
class MYDataset(Dataset):
    
    def __init__(self, X, Y, paths, transforms):
        self.X = X
        self.Y = Y
        self.paths = paths
        self.transforms = transforms
    
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, index):

        image = self.X[index]  

        if self.transforms:
            image = self.transforms(image=image)["image"]

        image = torch.tensor(image).unsqueeze(0)
        label = torch.tensor(self.Y[index]).unsqueeze(0)
        paths = self.paths[index]
        
        return image, label, paths
 
 # augs
my_transforms = A.Compose([
    # A.HorizontalFlip(p=p),
    # A.VerticalFlip(p=p),
    A.RandomBrightnessContrast(p=0.005),
    # A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.8, 0.8), p=p),
    A.Normalize(mean=0, std=1, max_pixel_value=255.0)
])
    

# read in synthetic test set
test_data_dir_syn_dir = Path('/Sythetic_patch_data_path')
test_data_syn_E_dir = test_data_dir_syn_dir / 'extrusion'
test_data_syn_C_dir = test_data_dir_syn_dir / 'control'

test_data_syn_E_paths = [f for f in test_data_syn_E_dir.glob("*.tif") if not f.name.startswith(".")]
test_data_syn_C_paths = [f for f in test_data_syn_C_dir.glob("*.tif") if not f.name.startswith(".")]


X_syn_test = []
Y_syn_test = []
P_syn_test = [] 

for i in (test_data_syn_E_paths, test_data_syn_C_paths):
    for f in i:
        
        # load 
        img = tiff.imread(f)
        img = img[ :, crop_val:-crop_val, crop_val:-crop_val]
        X_syn_test.append(img)
        P_syn_test.append(str(f))
        
        if i == test_data_syn_E_paths:
            Y_syn_test.append(1)
        else:
            Y_syn_test.append(0)
        

# data loader
Synth_test_dataset = MYDataset(X_syn_test, Y_syn_test, P_syn_test, my_transforms)

# making training and validation sets
dataset_size = len(Synth_test_dataset)

# data loader 
testing_loader = DataLoader(Synth_test_dataset, batch_size= 4)

# test model
all_preds = []
all_labels = []
incorrect= []

correct = 0 
total = 0

with torch.no_grad():
    for X, y, paths in testing_loader:
        
        X = X.to(device)
        y = y.to(device)
        
        outputs = zebnet(X)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        # accuracy
        correct += (preds == y).sum().item()
        total += y.numel()
        
        # store
        all_preds.append(probs.cpu())
        all_labels.append(y.cpu())

        #  store incorrect meta data 
        for i in range(len(preds)):
            if preds[i] != y[i]:
                incorrect.append({
                    "path": paths[i],
                    "true": int(y[i].item()),
                    "pred": int(preds[i].item()),
                    "prob": float(probs[i].item())
                })
                
                
accuracy = correct / total
print(f"synthetic test accuracy: {accuracy:.4f}")
#%% ROC, PRC and confusion matrix 
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

# ROC
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# PR Curve
precision, recall, _ = precision_recall_curve(all_labels, all_preds)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# confusion matrix 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# convert probabilities to binary predictions
threshold = 0.5
binary_preds = (all_preds > threshold).astype(int)

# confusion matrix
cm = confusion_matrix(all_labels, binary_preds)

tn, fp, fn, tp = cm.ravel()

total = cm.sum()


tn_pct = 100 * tn / total
fp_pct = 100 * fp / total
fn_pct = 100 * fn / total
tp_pct = 100 * tp / total

# plot
fig, ax = plt.subplots(figsize=(6,6))

ax.imshow(cm, cmap="Blues")

# labels for each square
labels = np.array([
    [f"True Negative\n{tn}\n({tn_pct:.1f}%)",
     f"False Positive\n{fp}\n({fp_pct:.1f}%)"],
    
    [f"False Negative\n{fn}\n({fn_pct:.1f}%)",
     f"True Positive\n{tp}\n({tp_pct:.1f}%)"]
])

for i in range(2):
    for j in range(2):

        # style
        text_color = 'white' if i == j else 'black'

        ax.text(
            j, i,
            labels[i, j],
            ha='center',
            va='center',
            fontsize=26,
            color=text_color
        )

# axis labels
ax.set_xticks([0,1])
ax.set_yticks([0,1])


plt.tight_layout()
plt.show()

#%% testing on synthetic tissue data
# paths
tissue_dir = Path('/')
output_dir = Path('./Model_output')
predicted_extrusion_dir = Path('./Preddicted_as_extrusion')
output_dir.mkdir(exist_ok=True)
predicted_extrusion_dir.mkdir(exist_ok=True)

tissue_paths = [f for f in tissue_dir.glob("*.tif") if not f.name.startswith(".")]

# params
patch_size = (16, 32, 32)
stride = (4, 8, 8)
pad_val = 0

pz, px, py = patch_size
sz, sx, sy = stride

zebnet.eval()

all_probs = []
all_results = []

for tissue_path in tissue_paths:

    print(f"Processing: {tissue_path.name}")

    volume = tiff.imread(tissue_path)
    volume = np.pad(volume, pad_val, mode= 'symmetric')

    # normalise full volume
    volume = volume/255.0

    Z, X, Y = volume.shape

    # grid size
    gz = (Z - pz) // sz + 1
    gx = (X - px) // sx + 1
    gy = (Y - py) // sy + 1

    pred_grid = np.zeros((gz, gx, gy), dtype=np.float32)

    # sliding window
    with torch.no_grad():
        for iz, z in enumerate(range(0, Z - pz + 1, sz)):
            for ix, x in enumerate(range(0, X - px + 1, sx)):
                for iy, y in enumerate(range(0, Y - py + 1, sy)):

                    patch = volume[z:z+pz, x:x+px, y:y+py]

                    patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch) + 1e-9)

                    patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                    output = zebnet(patch)
                    prob = torch.sigmoid(output).item()
                    
                    if prob >= 0.8:
                        patch_np = patch.detach().cpu().numpy()
                        tiff.imwrite(predicted_extrusion_dir / f"{tissue_path.stem}_extrusion_{iz}_{ix}_{iy}.tif", patch_np)
                        
                    
                    pred_grid[iz, ix, iy] = prob
                    
    name = tissue_path.stem
    tiff.imwrite(output_dir /f"{name}_prob_map.tif", pred_grid.astype(np.float32))
    print(f"saved probability map {name}")
    
    prob_volume = np.zeros_like(volume, dtype=np.float32)

    for iz, z in enumerate(range(0, Z - pz + 1, sz)):
        for ix, x in enumerate(range(0, X - px + 1, sx)):
            for iy, y in enumerate(range(0, Y - py + 1, sy)):
    
                prob = pred_grid[iz, ix, iy]
    
                prob_volume[z:z+pz, x:x+px, y:y+py] = np.maximum(
                    prob_volume[z:z+pz, x:x+px, y:y+py],
                    prob
                )
    
    # remove padding to match original image
    # prob_volume = prob_volume[pad_val:-pad_val, pad_val:-pad_val, pad_val:-pad_val]
    
    # save full-resolution map
    # tiff.imwrite(output_dir / f"{name}_prob_volume.tif", prob_volume.astype(np.float32))
    
    csv_path = tissue_dir / f"{name}_extrusions.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    
        probs_at_extrusions = []
        
        results_dir = predicted_extrusion_dir / "matched_to_extrusions"
        results_dir.mkdir(exist_ok=True)
        
        for _, row in df.iterrows():
            z, y, x = int(row[0]), int(row[1]), int(row[2])
        
            # map to nearest patch index
            iz = int(round((z - pz/2) / sz))
            ix = int(round((x - px/2) / sx))
            iy = int(round((y - py/2) / sy))
        
            # bounds check
            if (0 <= iz < pred_grid.shape[0] and
                0 <= ix < pred_grid.shape[1] and
                0 <= iy < pred_grid.shape[2]):
        
                prob = pred_grid[iz, iy, ix]
        
                # recover patch start coords
                z_start = iz * sz
                x_start = ix * sx
                y_start = iy * sy
        
                patch = volume[z_start:z_start+pz, y_start:y_start+py, x_start:x_start+px]
        
                # save patch
                save_name = (
                    f"{name}_z{z}_x{x}_y{y}_"
                    f"patch_{iz}_{ix}_{iy}_prob_{prob:.3f}.tif"
                )
        
                tiff.imwrite(results_dir / save_name, patch.astype(np.float32))
        
                # store in global results
                all_results.append({
                    "tissue": name,
                    "z": z,
                    "x": x,
                    "y": y,
                    "iz": iz,
                    "ix": ix,
                    "iy": iy,
                    "pred_prob": prob
                })
                
all_results_df = pd.DataFrame(all_results)
save_path = output_dir / "all_extrusion_predictions.csv"
all_results_df.to_csv(save_path, index=False)

#%% Testing on true data
from pathlib import Path
import numpy as np
import tifffile as tiff
import torch


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "model_output"
PREDICTED_EXTRUSION_DIR = DATA_DIR / "predicted_extrusions"

PATCH_SIZE = (16, 44, 44)
STRIDE = (4, 11, 11)

EXTRUSION_THRESHOLD = 0.5


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTED_EXTRUSION_DIR.mkdir(parents=True, exist_ok=True)

    tissue_paths = [
        f for f in DATA_DIR.glob("*.tif")
        if not f.name.startswith(".")
    ]

    if not tissue_paths:
        raise FileNotFoundError(f"No TIFF files found in {DATA_DIR}")

    tissue_path = tissue_paths[0]

    data = tiff.imread(tissue_path)

    pz, px, py = PATCH_SIZE
    sz, sx, sy = STRIDE

    zebnet.eval()

    T, _, _, _ = data.shape
    name = tissue_path.stem

    temp_files = []

    time_pbar = tqdm(range(T), desc="Processing timepoints")

    for t in time_pbar:

        time_pbar.set_postfix(current_t=t)

        volume = data[t].astype(np.float32) / 255.0

        Z, X, Y = volume.shape

        Zc = ((Z - pz) // sz) * sz + pz
        Xc = ((X - px) // sx) * sx + px
        Yc = ((Y - py) // sy) * sy + py

        volume = volume[:Zc, :Xc, :Yc]

        Z, X, Y = volume.shape

        gz = (Z - pz) // sz + 1
        gx = (X - px) // sx + 1
        gy = (Y - py) // sy + 1

        pred_grid = np.zeros((gz, gx, gy), dtype=np.float32)

        total_patches = gz * gx * gy

        patch_pbar = tqdm(
            total=total_patches,
            desc=f"Patches t={t}",
            leave=False
        )

        with torch.no_grad():

            for iz, z in enumerate(range(0, Z - pz + 1, sz)):
                for ix, x in enumerate(range(0, X - px + 1, sx)):
                    for iy, y in enumerate(range(0, Y - py + 1, sy)):

                        patch = volume[
                            z:z + pz,
                            x:x + px,
                            y:y + py
                        ]

                        patch_tensor = (
                            torch.tensor(
                                patch,
                                dtype=torch.float32
                            )
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(device)
                        )

                        output = zebnet(patch_tensor)
                        prob = torch.sigmoid(output).item()

                        if prob >= EXTRUSION_THRESHOLD:

                            tiff.imwrite(
                                PREDICTED_EXTRUSION_DIR /
                                f"{name}_t{t}_extrusion_{iz}_{ix}_{iy}.tif",
                                patch.astype(np.float32)
                            )

                        pred_grid[iz, ix, iy] = prob

                        patch_pbar.update(1)

        patch_pbar.close()

        temp_path = OUTPUT_DIR / f"{name}_t{t}_prob_map.tif"

        tiff.imwrite(
            temp_path,
            pred_grid.astype(np.float32)
        )

        temp_files.append(temp_path)

    print("\nCombining all timepoints into 4D stack...")

    all_preds = []

    for temp_path in tqdm(temp_files, desc="Combining TIFFs"):
        all_preds.append(tiff.imread(temp_path))

    all_preds = np.stack(all_preds, axis=0)

    final_path = OUTPUT_DIR / f"{name}_FULL_prob_map_4D.tif"

    tiff.imwrite(
        final_path,
        all_preds.astype(np.float32)
    )

    print(f"\nSaved final 4D stack to:\n{final_path}")


if __name__ == "__main__":
    main()