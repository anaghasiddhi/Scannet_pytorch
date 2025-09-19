from torch.nn.utils.rnn import pad_sequence
from scannet_pytorch.utilities.io_utils import to_tensor

def process_inputs(inputs_batch, device):
    transposed = list(zip(*inputs_batch))
    processed = []

    for i, inputs_of_one_type in enumerate(transposed):
        try:
            tensor_list = [to_tensor(x, device) for x in inputs_of_one_type]

            if len(set(t.shape for t in tensor_list)) == 1:
                stacked = torch.stack(tensor_list)
            elif all(t.ndim == 2 for t in tensor_list):  # (L, F)
                stacked = pad_sequence(tensor_list, batch_first=True)
            elif all(t.ndim == 1 for t in tensor_list):  # (L,)
                stacked = pad_sequence(tensor_list, batch_first=True)
            else:
                print(f"[SKIP] Unhandled shape in group {i}")
                return None

            processed.append(stacked)

        except Exception as e:
            shapes = [x.shape if hasattr(x, "shape") else type(x) for x in inputs_of_one_type]
            print(f"[SKIP] Cannot stack input group {i} due to shape mismatch: {shapes}")
            return None

    return processed

from torch.nn.utils.rnn import pad_sequence
import torch

def filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    inputs, labels = zip(*batch)

    labels = [l if isinstance(l, torch.Tensor) else torch.from_numpy(l) for l in labels]
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    return list(inputs), labels
def dry_run_dataset(dataset, max_checks=20):
    print("🔎 Running dataset dry run check...")
    count = 0
    for i in range(len(dataset)):
        try:
            example = dataset[i]
            if example is not None:
                count += 1
            if count >= 1:
                print(f"✅ Dry run succeeded. Found at least one usable example at index {i}.")
                return True
        except Exception as e:
            print(f"[⚠️] Error at index {i}: {e}")
        if i >= max_checks:
            break
    print("❌ Dry run failed: No usable examples found in dataset.")
    return False

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score):
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
