import json

import numpy as np
import onnxruntime as ort
import torch

from train import LaneNet

CKPT = 'models/lane_best.pt'  # run from the repo root
OUT = 'models/lane_best.onnx'
H, W = 49, 108

ckpt = torch.load(CKPT, map_location='cpu')

model = LaneNet()
dummy = torch.zeros(1, 3, H, W)
model(dummy)
model.load_state_dict(ckpt['model'])
model.eval()

torch.onnx.export(
  model, dummy, OUT,
  input_names=['image'], output_names=['steer_norm'],
  dynamic_axes={'image': {0: 'batch'}, 'steer_norm': {0: 'batch'}},
  opset_version=17, dynamo=False,
)

meta = {
  'y_mean': ckpt['y_mean'],
  'y_std': ckpt['y_std'],
  'scale': ckpt.get('scale', 0.5),
  'input_hw': [H, W]
}

with open(OUT.replace('.onnx', '.json'), 'w') as f:
  json.dump(meta, f, indent=2)

# parity check
x = torch.rand(8, 3, H, W)
with torch.no_grad():
  ref = model(x).numpy()
sess = ort.InferenceSession(OUT, providers=['CPUExecutionProvider'])
out = sess.run(None, {'image': x.numpy()})[0].reshape(-1)
print('max abs diff torch vs onnx:', np.max(np.abs(ref - out)))
print('wrote', OUT, 'and', OUT.replace('.onnx', '.json'), meta)
