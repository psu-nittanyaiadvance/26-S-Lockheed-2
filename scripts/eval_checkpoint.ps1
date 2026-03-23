param(
    [string]$Checkpoint = "checkpoints\checkpoint_epoch1.pth",
    [string]$ImgDir = "datasets\FilteredSouthAsia\HandLabeled\S1Hand",
    [string]$MaskDir = "datasets\FilteredSouthAsia\HandLabeled\S1OtsuLabelHand",
    [int]$Classes = 1,
    [int]$BatchSize = 4,
    [int]$PatchSize = 256,
    [double]$Overlap = 0.2,
    [switch]$Amp
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).ProviderPath
$CheckpointPath = (Resolve-Path -LiteralPath $Checkpoint).ProviderPath
$ImgDirPath = (Resolve-Path -LiteralPath $ImgDir).ProviderPath
$MaskDirPath = (Resolve-Path -LiteralPath $MaskDir).ProviderPath

$ampArgs = @()
if ($Amp.IsPresent) { $ampArgs = @("--amp") }

@'
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def tversky_loss(inputs, targets, alpha=0.7, beta=0.3, epsilon=1e-6):
    inputs = torch.sigmoid(inputs) if inputs.shape[1] == 1 else F.softmax(inputs, dim=1)
    inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
    targets = targets.view(targets.shape[0], targets.shape[1], -1)
    tp = (inputs * targets).sum(dim=2)
    fp = ((1 - targets) * inputs).sum(dim=2)
    fn = (targets * (1 - inputs)).sum(dim=2)
    tversky = (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon)
    return 1 - tversky.mean()


def dict_collate(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    masks = torch.stack([b[1] for b in batch], dim=0)
    return {"image": images, "mask": masks}


parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--img-dir", required=True)
parser.add_argument("--mask-dir", required=True)
parser.add_argument("--classes", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--patch-size", type=int, default=256)
parser.add_argument("--overlap", type=float, default=0.2)
parser.add_argument("--amp", action="store_true", default=False)
args = parser.parse_args()

repo_root = Path(args.repo_root)
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from data_loader import SARDataset, PatchDataset
from eval import evaluate
from UNet.UNetModel import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=2, n_classes=args.classes, bilinear=False)

state = torch.load(args.checkpoint, map_location=device)
state.pop("mask_values", None)
model.load_state_dict(state)
model.to(device)

img_dir = Path(args.img_dir)
mask_dir = Path(args.mask_dir)
ids = sorted({p.stem for ext in ("*.tif", "*.tiff") for p in img_dir.glob(ext)})
if not ids:
    raise RuntimeError("No .tif/.tiff files found in the image directory")

ds = SARDataset(
    img_root=img_dir,
    mask_root=mask_dir,
    ids_or_paths=ids,
    mode="strong",
    mask_id_suffix_map={"S1Hand": "S1OtsuLabelHand"},
    expected_img_bands=2,
)
ds.mask_values = [0, 1]

patches = PatchDataset(ds, patch_size=args.patch_size, overlap=args.overlap)
loader = DataLoader(
    patches,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=dict_collate,
)

metrics = evaluate(model, loader, device, amp=args.amp, criterion=tversky_loss, n_classes=args.classes)
print(metrics)
'@ | python - --repo-root "$RepoRoot" --checkpoint "$CheckpointPath" --img-dir "$ImgDirPath" --mask-dir "$MaskDirPath" --classes $Classes --batch-size $BatchSize --patch-size $PatchSize --overlap $Overlap @ampArgs
