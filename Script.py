
!pip install monai torchio nibabel scikit-image tqdm matplotlib

import os
import shutil
import tarfile
import random
import gc
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchio as tio
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.utils import one_hot

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import measure


# CONFIG / HYPERPARAMS

EXTRACTED_DIR = "/kaggle/working/BraTS2021/"        # folder containing patient subfolders (after extraction)
PREPROC_DIR = "/kaggle/working/preprocessed/"       # intermediate npz outputs
PREPROC_FIXED_DIR = "/kaggle/working/preprocessed_fixed/"
os.makedirs(PREPROC_DIR, exist_ok=True)
os.makedirs(PREPROC_FIXED_DIR, exist_ok=True)

TARGET_SHAPE = (96, 96, 96)    # final cubic shape
NUM_PATIENTS = 800            
CHANNELS = ["t1ce", "flair"]   
BATCH_SIZE = 1                 
NUM_WORKERS = 2
EPOCHS = 1                     
LR = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def safe_extract(tar_path, out_dir):
    def _is_secure(member, target_dir):
        abs_target = os.path.abspath(target_dir)
        abs_member = os.path.abspath(os.path.join(target_dir, member.name))
        return abs_member.startswith(abs_target)

    with tarfile.open(tar_path, "r") as tar:
        for m in tar.getmembers():
            if not _is_secure(m, out_dir):
                raise RuntimeError("Unsafe path in tar file")
        tar.extractall(path=out_dir)

safe_extract("/kaggle/input/brats-2021-task1/BraTS2021_Training_Data.tar", EXTRACTED_DIR)


# patients dictionary
def collect_patients(root_dir):
    patients = {}
    required_modalities = ["t1", "t1ce", "t2", "flair", "seg"]
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root folder not found: {root_dir}")
    for entry in sorted(os.listdir(root_dir)):
        ppath = os.path.join(root_dir, entry)
        if not os.path.isdir(ppath):
            continue
        mod_dict = {m: None for m in required_modalities}
        for f in os.listdir(ppath):
            lf = f.lower()
            if not lf.endswith(".nii.gz"):
                continue
            fp = os.path.join(ppath, f)
            
            if "_t1ce.nii.gz" in lf:
                mod_dict["t1ce"] = fp
            elif "_t1.nii.gz" in lf and "_t1ce" not in lf:
                mod_dict["t1"] = fp
            elif "_t2.nii.gz" in lf:
                mod_dict["t2"] = fp
            elif "_flair.nii.gz" in lf:
                mod_dict["flair"] = fp
            elif "_seg.nii.gz" in lf:
                mod_dict["seg"] = fp
        if all(mod_dict[m] is not None for m in required_modalities):
            patients[entry] = mod_dict
        else:
        
            missing = [m for m in required_modalities if mod_dict[m] is None]
            print(f"[WARN] skipping {entry} - missing: {missing}")
    return patients

patients = collect_patients(EXTRACTED_DIR)
if len(patients) == 0:
    raise RuntimeError("No valid patient folders found in EXTRACTED_DIR; check path and extraction.")
print(f"[INFO] Found {len(patients)} patients")


all_ids = list(patients.keys())
if NUM_PATIENTS is not None:
    random.shuffle(all_ids)
    selected_ids = all_ids[:NUM_PATIENTS]
else:
    selected_ids = all_ids

print(f"[INFO] Selected {len(selected_ids)} patients for preprocessing.")


# Preprocessing helpers
def load_nii_as_np(path):
    """Load NIfTI -> numpy float32"""
    return nib.load(path).get_fdata().astype(np.float32)

def zscore_nonzero(volume):
    mask = volume != 0
    if mask.sum() == 0:
        # nothing to normalize
        return volume
    vals = volume[mask]
    m = vals.mean()
    s = vals.std()
    if s < 1e-6:
        return volume - m
    return (volume - m) / (s + 1e-6)

def safe_get_joint_bbox(volumes):
    """
    Compute joint bbox across volumes.
    If the mask is empty (very unlikely), return a central crop bbox.
    """
    mask = np.zeros_like(volumes[0], dtype=bool)
    for v in volumes:
        mask |= (v != 0)
    if not mask.any():
        vol_shape = volumes[0].shape
        minc = np.maximum(0, np.array(vol_shape)//2 - np.array(TARGET_SHAPE)//2)
        maxc = np.minimum(np.array(vol_shape)-1, minc + np.array(TARGET_SHAPE) - 1)
        return minc, maxc
    coords = np.array(np.nonzero(mask))
    minc = coords.min(axis=1)
    maxc = coords.max(axis=1)
    return minc, maxc

def crop_to_bbox(vol, minc, maxc):
    return vol[minc[0]:maxc[0]+1, minc[1]:maxc[1]+1, minc[2]:maxc[2]+1]

def resize_volume_trilinear(vol, out_shape):
    t = torch.tensor(vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
    out = F.interpolate(t, size=out_shape, mode="trilinear", align_corners=False)
    return out.squeeze().numpy()

def resize_label_nearest(seg, out_shape):
    t = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size=out_shape, mode="nearest")
    return out.squeeze().numpy().astype(np.uint8)

def remap_brats_labels(seg):
    """
    Map original BraTS labels {0,1,2,4} to our working labels {0,1,2} where:
      0 -> background (and also keep 1->0 per earlier mapping if desired)
    We'll follow your notebook's mapping:
      original 1 -> 0 (necrotic core)
      original 2 -> 1 (edema)
      original 4 -> 2 (enhancing tumor)
    Any unexpected label values will be mapped to 0 (background).
    """
    out = np.zeros_like(seg, dtype=np.uint8)
    out[seg == 1] = 0
    out[seg == 2] = 1
    out[seg == 4] = 2
    return out


# Preprocessing
print("[INFO] Starting preprocessing...")
for pid in tqdm(selected_ids, desc="Preprocessing"):
    try:
        p = patients[pid]
        loaded_channels = []
        for ch in CHANNELS:
            loaded_channels.append(load_nii_as_np(p[ch]))
        seg = load_nii_as_np(p["seg"]).astype(np.uint8)

        minc, maxc = safe_get_joint_bbox(loaded_channels + [seg])
        loaded_channels = [crop_to_bbox(v, minc, maxc) for v in loaded_channels]
        seg = crop_to_bbox(seg, minc, maxc)

        loaded_channels = [zscore_nonzero(v) for v in loaded_channels]

        resized_chs = [resize_volume_trilinear(v, TARGET_SHAPE) for v in loaded_channels]
        resized_seg = resize_label_nearest(seg, TARGET_SHAPE)

        image = np.stack(resized_chs, axis=0).astype(np.float32)

        out_path = os.path.join(PREPROC_DIR, f"{pid}.npz")
        np.savez_compressed(out_path, image=image.astype(np.float16), seg=resized_seg.astype(np.uint8))

    except Exception as e:
        print(f"[ERROR] Preprocessing {pid}: {e}")
        continue

print("[INFO] Preprocessing finished.")

# Fixing labels
print("[INFO] Fixing label maps and saving to:", PREPROC_FIXED_DIR)
files = sorted([f for f in os.listdir(PREPROC_DIR) if f.endswith(".npz")])
for fn in tqdm(files, desc="Fixing labels"):
    src = os.path.join(PREPROC_DIR, fn)
    dst = os.path.join(PREPROC_FIXED_DIR, fn)
    try:
        npz = np.load(src, allow_pickle=False)
        image = npz["image"].astype(np.float16)
        seg = npz["seg"].astype(np.uint8)
        uniques = np.unique(seg)
        if not np.all(np.isin(uniques, [0,1,2])):
            seg = remap_brats_labels(seg)
        np.savez_compressed(dst, image=image, seg=seg)
    except Exception as e:
        print(f"[ERROR] fixing {fn}: {e}")

fixed_files = sorted([f for f in os.listdir(PREPROC_FIXED_DIR) if f.endswith(".npz")])
if len(fixed_files) == 0:
    raise RuntimeError("No fixed preprocessed files found - aborting.")
print("[INFO] Number of preprocessed-fixed files:", len(fixed_files))


# Dataset class
class BraTSDataset(Dataset):
    """
    Loads preprocessed .npz with:
        image: (C, X, Y, Z) float16 or float32
        seg:   (X, Y, Z) uint8
    Returns TorchIO Subject style (but as tensors) so transforms can be applied elsewhere.
    """
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        path = os.path.join(self.data_dir, fn)
        npz = np.load(path, allow_pickle=False)
        image = npz["image"].astype(np.float32)  
        seg = npz["seg"].astype(np.uint8)        
        return {
            "id": fn.replace(".npz", ""),
            "image": image,
            "seg": seg
        }


# Creating dataset + split
full_dataset = BraTSDataset(PREPROC_FIXED_DIR)
n_total = len(full_dataset)
train_size = int(0.7 * n_total)
val_size = int(0.15 * n_total)
test_size = n_total - train_size - val_size
train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(SEED))
print(f"[INFO] splits - train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")


train_transforms = tio.Compose([
    tio.RandomFlip(axes=('LR',), p=0.5),
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5, p=0.5),
    tio.RandomElasticDeformation(num_control_points=7, max_displacement=7, p=0.3),
    tio.RandomNoise(p=0.2),
    tio.RandomBlur(p=0.2),
    tio.ZNormalization(masking_method=None), 
    tio.CropOrPad(TARGET_SHAPE)
])

val_transforms = tio.Compose([
    tio.ZNormalization(masking_method=None),
    tio.CropOrPad(TARGET_SHAPE)
])

test_transforms = val_transforms


class TransformWrapper(Dataset):
    def __init__(self, base_subset, transforms, mode='train'):
        self.base = base_subset   
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]  
        subject_dict = {}
        subject_dict["image"] = tio.ScalarImage(tensor=torch.tensor(sample["image"], dtype=torch.float32))
        if sample.get("seg", None) is not None:
            seg_tensor = torch.tensor(sample["seg"], dtype=torch.int64).unsqueeze(0) 
            subject_dict["seg"] = tio.LabelMap(tensor=seg_tensor)
        subject = tio.Subject(**subject_dict)
        if self.transforms is not None:
            subject = self.transforms(subject)
        out = {"id": sample["id"], "image": subject["image"].data} 
        if "seg" in subject:
            out["seg"] = subject["seg"].data 
        else:
            out["seg"] = None
        return out

train_dataset = TransformWrapper(train_ds, train_transforms, mode='train')
val_dataset = TransformWrapper(val_ds, val_transforms, mode='val')
test_dataset = TransformWrapper(test_ds, test_transforms, mode='test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


sample = next(iter(train_loader))
print("[SANITY] image shape:", sample["image"].shape, "seg shape:", sample["seg"].shape if sample["seg"] is not None else None)


# Model, loss, optimizer
model = UNet(
    spatial_dims=3,
    in_channels=len(CHANNELS),
    out_channels=3,   
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_function = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    squared_pred=True,
    include_background=False
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

dice_metric = DiceMetric(include_background=False, reduction="mean")

hausdorff_metric = HausdorffDistanceMetric(
    include_background=False,
    percentile=95,
    reduction="mean"
)



print("[INFO] Model & loss ready. Device:", device)

# Training loop with validation 
scaler = GradScaler()
best_val_dice = -1.0
checkpoint_path = "/kaggle/working/best_model.pth"


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train")
    for batch in pbar:
        imgs = batch["image"].to(device)           
        segs = batch["seg"].to(device).long()      

        if torch.any(segs < 0) or torch.any(segs > 2):
            raise ValueError("Labels outside expected range {0,1,2}")

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(imgs)  
            loss = loss_function(outputs, segs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(train_loader)
    print(f"[TRAIN] Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    dice_metric.reset()
    hausdorff_metric.reset()
    with torch.no_grad():
        pbarv = tqdm(val_loader, desc=f"Epoch {epoch+1} Val")
        for batch in pbarv:
            imgs = batch["image"].to(device)
            segs = batch["seg"].to(device).long()  
            with autocast():
                out = model(imgs)  

            preds = torch.argmax(out, dim=1, keepdim=True) 
            segs_oh = one_hot(segs, num_classes=3)
            preds_oh = one_hot(preds, num_classes=3)
            dice_metric(y_pred=preds_oh, y=segs_oh)
            hausdorff_metric(y_pred=preds_oh, y=segs_oh)

    try:
        val_dice = dice_metric.aggregate().item()
    except Exception:
        val_dice = float(dice_metric.aggregate())  # fallback

    hd_vals = hausdorff_metric.aggregate()
    if isinstance(hd_vals, torch.Tensor):
        hd_vals = hd_vals.cpu().numpy().tolist()
    else:
        pass

    print(f"[VAL] Epoch {epoch+1} Dice (mean over classes): {val_dice:.4f}")
    print(f"[VAL] Epoch {epoch+1} Hausdorff95 per-class (classes 0..2): {hd_vals}")


    # best checkpoint
    if val_dice > best_val_dice:
        best_val_dice = val_dice

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_dice": val_dice,
        }
    
        torch.save(checkpoint, checkpoint_path)
        print(f"[CHECKPOINT] Saved best model (val_dice={val_dice:.4f}) to {checkpoint_path}")


    dice_metric.reset()
    hausdorff_metric.reset()

print("[TRAINING COMPLETE] best_val_dice:", best_val_dice)


# Test-time prediction + Dice & Hausdorff on test set

if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[INFO] Loaded checkpoint from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

model.eval()
dice_metric.reset()
hausdorff_metric.reset()
test_dice_scores = []
test_hd_scores = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        imgs = batch["image"].to(device)
        segs = batch["seg"].to(device).long()

        out = model(imgs)
        preds = torch.argmax(out, dim=1, keepdim=True)
        segs_oh = one_hot(segs, num_classes=3)
        preds_oh = one_hot(preds, num_classes=3)
        dice_metric(y_pred=preds_oh, y=segs_oh)
        hausdorff_metric(y_pred=preds_oh, y=segs_oh)

dice_test = dice_metric.aggregate().item()
hd_test = hausdorff_metric.aggregate()
if isinstance(hd_test, torch.Tensor):
    hd_test = hd_test.cpu().numpy().tolist()

print(f"[TEST] Mean Dice: {dice_test:.4f}")
print(f"[TEST] Hausdorff95 per-class: {hd_test}")


# Visualization helpers
def visualize_slice(image_tensor, seg_tensor=None, pred_tensor=None, slice_idx=None, channel_idx=0):
    """
    image_tensor: (C, X, Y, Z) torch tensor or numpy
    seg_tensor: (1, X, Y, Z) or numpy
    pred_tensor: (1, X, Y, Z) or logits/tensor -> will convert via argmax if needed
    """
    if isinstance(image_tensor, torch.Tensor):
        img = image_tensor.cpu().numpy()
    else:
        img = np.asarray(image_tensor)
    if seg_tensor is not None:
        seg = seg_tensor.cpu().numpy().squeeze(0)
    else:
        seg = None
    if pred_tensor is not None:
        if pred_tensor.ndim == 5 and pred_tensor.shape[1] > 1:
            p = torch.argmax(pred_tensor, dim=1, keepdim=True).cpu().numpy().squeeze(0)
        else:
            p = pred_tensor.cpu().numpy().squeeze(0)
    else:
        p = None

    if slice_idx is None:
        slice_idx = img.shape[-1] // 2

    channel_slice = img[channel_idx, :, :, slice_idx]
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(channel_slice, cmap='gray')
    plt.title("Input channel")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    if seg is not None:
        plt.imshow(channel_slice, cmap='gray')
        plt.imshow(seg[:, :, slice_idx], cmap='jet', alpha=0.5)
        plt.title("Ground truth")
        plt.axis('off')
    else:
        plt.title("No GT")
        plt.axis('off')

    plt.subplot(1, 3, 3)
    if p is not None:
        plt.imshow(channel_slice, cmap='gray')
        plt.imshow(p[:, :, slice_idx], cmap='jet', alpha=0.5)
        plt.title("Prediction")
        plt.axis('off')
    else:
        plt.title("No pred")
        plt.axis('off')

    plt.show()

def render_3d_surface(seg_volume, class_id=2, spacing=(1.0,1.0,1.0)):
    """
    Render a 3D triangular mesh (vertices, faces) for a given class using marching cubes.
    seg_volume: numpy array (X,Y,Z)
    Returns vertices, faces for potential plotting or saving.
    """
    mask = (seg_volume == class_id).astype(np.uint8)
    if mask.sum() == 0:
        print(f"[WARN] No voxels for class {class_id} in provided volume.")
        return None, None
    verts, faces, normals, values = measure.marching_cubes(mask, level=0.5, spacing=spacing)
    return verts, faces

