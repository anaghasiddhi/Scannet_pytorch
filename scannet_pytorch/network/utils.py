import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_to_value(tensor, value):
    with torch.no_grad():
        tensor.copy_(torch.tensor(value, dtype=torch.float32))


def constraint_between(w, minimum=-1.0, maximum=1.0):
    return torch.clamp(w, min=minimum, max=maximum)


def fixed_norm(w, value=1.0, axis=0):
    norm = torch.norm(w, dim=axis, keepdim=True) + 1e-7
    return w * (value / norm)


def symmetric(w):
    return (w + w.permute(1, 0, 2)) / 2


def masked_categorical_cross_entropy(y_true, y_pred):
    # Assume y_true is one-hot or soft labels
    mask = torch.max((y_true > 0).float(), dim=-1).values
    loss = -(y_true * torch.log_softmax(y_pred, dim=-1)).sum(dim=-1)
    return (loss * mask).mean() / (mask.mean() + 1e-8)


def masked_categorical_accuracy(y_true, y_pred):
    # mask: whether each sample has a valid label
    mask = torch.max((y_true > 0).float(), dim=-1).values
    y_true_class = torch.argmax(y_true, dim=-1)
    y_pred_class = torch.argmax(y_pred, dim=-1)
    correct = (y_true_class == y_pred_class).float()
    return (correct * mask).sum() / (mask.sum() + 1e-8)


def masked_MSE(y_true, y_pred):
    mask = (y_true != -1).float()
    squared_error = (y_pred - y_true) ** 2
    masked_se = squared_error * mask
    return masked_se.mean() / (mask.mean() + 1e-8)


class MaxAbsPooling(nn.Module):
    def __init__(self, axis=-1):
        super(MaxAbsPooling, self).__init__()
        self.axis = axis

    def forward(self, x):
        ndim = x.dim()
        axis = self.axis if self.axis >= 0 else self.axis + ndim

        if axis != ndim - 1:
            perm = list(range(ndim))
            perm[axis], perm[-1] = perm[-1], perm[axis]
            x = x.permute(*perm)

        abs_x = torch.abs(x)
        idx = torch.argmax(abs_x, dim=-1, keepdim=True)
        out = torch.gather(x, dim=-1, index=idx).squeeze(-1)

        return out


def embeddings_initializer(shape, dtype=torch.float32):
    if shape[0] == shape[1] + 1:
        init = np.zeros(shape, dtype=np.float32)
        for i in range(1, shape[0]):
            init[i, i - 1] = np.sqrt(shape[1])
    else:
        init = np.random.randn(*shape).astype(np.float32)
        init[0, :] *= 0
        init[1:, :] /= np.sqrt((init[1:, :]**2).mean(0))[np.newaxis, :]
    return torch.tensor(init, dtype=dtype)


def scatter_sum(src, index, dim=1, dim_size=None):
    """
    Scatter `src` values into a zero tensor of size `dim_size` along `dim`,
    summing values at each index. Handles out-of-bound indices by clamping.

    Args:
        src (Tensor): The source tensor to scatter.
        index (Tensor): The index tensor indicating where to scatter to.
        dim (int): The dimension along which to scatter.
        dim_size (int): The size of the output tensor along `dim`.

    Returns:
        Tensor: A tensor with scattered sums.
    """
    if dim_size is None:
        raise ValueError("`dim_size` must be provided to allocate the output tensor.")

    out_size = list(src.shape)
    out_size[dim] = dim_size
    out = torch.zeros(out_size, device=src.device, dtype=src.dtype)

    # Clamp indices to valid range and check for out-of-bounds
    max_valid = dim_size - 1
    if index.max() > max_valid or index.min() < 0:
#         print(f"[⚠️] Clamping indices: detected values outside 0–{max_valid}.")
        index = index.clamp(0, max_valid)

    # Expand index shape to match src if needed
    while index.ndim < src.ndim:
        index = index.unsqueeze(-1)
    if index.shape != src.shape:
        try:
            index = index.expand_as(src)
        except RuntimeError as e:
            raise RuntimeError(f"[❌] scatter_sum shape mismatch: src {src.shape}, index {index.shape}. Cannot expand.") from e

    index = index.long()
    return out.scatter_add_(dim, index, src)

class PoolbyIndices(nn.Module):
    def __init__(self):
        super(PoolbyIndices, self).__init__()

    def forward(self, inputs, masks=[None, None, None]):
        index_target, index_source, array_source = inputs  # Shapes: [B, Nt, 1], [B, Ns, 1], [B, Ns, D]
        mask0, mask1, _ = masks

        # [B, Nt, Ns] boolean: match[i,j,k] = index_target[i,j,0] == index_source[i,k,0]
        mapping = (index_target[:, :, 0].unsqueeze(-1) == index_source[:, :, 0].unsqueeze(-2))

        if mask0 is not None:
            mapping = mapping & mask0.unsqueeze(-1).bool()
        if mask1 is not None:
            mapping = mapping & mask1.unsqueeze(-2).bool()

        mapping = mapping.float()  # Convert to float for math

        # Weighted mean pooling
        weights_sum = mapping.sum(dim=-1, keepdim=True) + 1e-10
        array_target = torch.bmm(mapping, array_source) / weights_sum

        return array_target
