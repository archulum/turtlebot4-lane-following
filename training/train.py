import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Folder containing the recorded runs (fwd_1_*, rev_1_*, ...). Override with LANE_DATASET=/path/to/lane_dataset
ROOT = os.path.expanduser(os.environ.get('LANE_DATASET', 'data/lane_dataset'))
TRAIN_TAGS = ['fwd_1', 'fwd_3', 'rev_1', 'rev_3']
VAL_TAGS = ['fwd_2', 'rev_2']
SCALE = 0.5
EPOCHS = 20
BATCH = 64
LR = 1e-3
CKPT = 'models/lane_best.pt'  # run from the repo root

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

## Data prep function
def load_runs(tags, scale=SCALE):
  X, y = [], []
  for tag in tags:
    runs = glob.glob(f'{ROOT}/{tag}_*')
    assert len(runs) == 1, f'{tag}: expected 1 folder, found {runs}'
    run = runs[0]
    df = pd.read_csv(f'{run}/labels.csv')
    for fn in df.frame:
      img = cv2.imread(f'{run}/scenes/cropped/{fn}')
      assert img is not None, f'could not read {run}/scenes/cropped/{fn}'
      img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
      X.append(img[:, :, ::-1].copy())
    y.append(df.angular_z.values.astype(np.float32))
    print(f'{tag}: {len(df)} frames, angular_z mean={df.angular_z.mean():+.3f}')

  X = torch.from_numpy(np.stack(X)).permute(0, 3, 1, 2).contiguous()
  y = torch.from_numpy(np.concatenate(y))
  return X, y

# model
class LaneNet(nn.Module):
  def __init__(self):
    super().__init__()

    def block(i, o):
      return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1),
                           nn.BatchNorm2d(o), nn.ReLU()
                           )

    self.features = nn.Sequential(block(3, 16), block(16, 32),
                                  block(32, 64), block(64, 64))
    self.head = nn.Sequential(nn.Flatten(), nn.LazyLinear(64),
                              nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

  def forward(self, x):
    return self.head(self.features(x)).squeeze(1)

def prep(x, train=False):
  x = x.to(device).float() / 255
  if train:
    b = (torch.rand(x.size(0), 1, 1, 1, device=device) - 0.5) * 0.3
    c = 1 + (torch.rand(x.size(0), 1, 1, 1, device=device) - 0.5) * 0.3
    x = ((x - 0.5) * c + 0.5 + b).clamp(0, 1)
  return x

def main():
  print(f'device: {device}')
  Xtr, ytr = load_runs(TRAIN_TAGS)
  Xval, yval = load_runs(VAL_TAGS)
  print('train', tuple(Xtr.shape), 'val', tuple(Xval.shape))

  # normalize target to roughly unit scale
  y_mean, y_std = ytr.mean().item(), ytr.std().item()
  ytr_n = (ytr - y_mean) / y_std
  yval_n = (yval - y_mean) / y_std

  # data loaders
  train_dl = DataLoader(TensorDataset(Xtr, ytr_n), batch_size=BATCH, shuffle=True)
  val_dl = DataLoader(TensorDataset(Xval, yval_n), batch_size=128)

  model = LaneNet().to(device)
  model(prep(Xtr[:2])) #materialize lazy layer before building optimiser
  opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

  def evaluate():
    model.eval()
    tot = 0.0
    with torch.no_grad():
      for x, y in val_dl:
        tot += F.mse_loss(model(prep(x)), y.to(device), reduction='sum')
    return tot / len(yval_n)

  baseline = (yval_n ** 2).mean().item()
  print(f'baseline val MSE (predict train mean): {baseline:.6f}')

  best = float('inf')
  for epoch in range(EPOCHS):
    model.train()
    tot = 0.0
    for x, y in train_dl:
      opt.zero_grad()
      loss = F.mse_loss(model(prep(x, train=True)), y.to(device))
      loss.backward()
      opt.step()
      tot += loss.item() * len(y)
    sched.step()
    val = evaluate()
    rmse = (val ** 0.5) * y_std
    print(f'epoch {epoch+1:2d} train {tot / len(ytr_n):.6f} val {val:.6f} rmse {rmse:.6f}')
    if val < best:
      best = val
      torch.save({'model': model.state_dict(),
                  'y_mean': y_mean,
                  'y_std': y_std,
                  'scale': SCALE}, CKPT)

  print(f'best val MSE {best:.6f} (baseline {baseline:.6f}), saved to {CKPT}')

if __name__ == '__main__':
  main()

