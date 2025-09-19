import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture


class GaussianKernel(nn.Module):
    def __init__(self, N, initial_values, covariance_type='diag', eps=1e-1):
        super(GaussianKernel, self).__init__()
        self.N = N
        self.eps = eps
        self.covariance_type = covariance_type
        assert self.covariance_type in ['diag', 'full']

        centers_init, widths_or_precision = initial_values
        d = centers_init.shape[0]

        self.centers = nn.Parameter(torch.tensor(centers_init, dtype=torch.float32))  # [d, N]

        if self.covariance_type == 'diag':
            self.widths = nn.Parameter(torch.tensor(widths_or_precision, dtype=torch.float32))  # [d, N]
        elif self.covariance_type == 'full':
            self.sqrt_precision = nn.Parameter(torch.tensor(widths_or_precision, dtype=torch.float32))  # [d, d, N]

    def forward(self, x):
        # x shape: [B, ..., d]
        d = x.shape[-1]
        x_exp = x.unsqueeze(-1)  # [B, ..., d, 1]
        center_shape = [1] * (x.dim() - 1) + list(self.centers.shape)  # [1, ..., d, N]
        centers = self.centers.view(*center_shape)

        if self.covariance_type == 'diag':
            widths = self.widths.view(*center_shape)
            diff = (x_exp - centers) / (self.eps + widths)
            dist_sq = (diff ** 2).sum(dim=-2)  # Sum over d
        elif self.covariance_type == 'full':
            # [B, ..., d, 1] - [1, ..., d, N] = [B, ..., d, N]
            diff = x_exp - centers  # [B, ..., d, N]
            sqrt_precision = self.sqrt_precision.unsqueeze(0)  # [1, d, d, N]
            diff_exp = diff.unsqueeze(-3)  # [B, ..., 1, d, N]
            sqrt_precision_exp = sqrt_precision  # [1, d, d, N]
            prod = (diff_exp * sqrt_precision_exp).sum(dim=-2)  # [B, ..., d, N]
            dist_sq = (prod ** 2).sum(dim=-2)  # [B, ..., N]

        activity = torch.exp(-0.5 * dist_sq)
        return activity

def inv_root_matrix(H):
    lam, v = np.linalg.eigh(H)
    return np.dot(v,  1 / np.sqrt(lam)[:, np.newaxis] * v.T)


def initialize_GaussianKernel(points, N, covariance_type='diag', reg_covar=1e-1, n_init=10):
    GMM = GaussianMixture(n_components=N,
                          covariance_type=covariance_type,
                          reg_covar=reg_covar,
                          n_init=n_init,
                          verbose=1)
    GMM.fit(points)

    centers = GMM.means_
    covariances = GMM.covariances_
    probas = GMM.weights_

    order = np.argsort(probas)[::-1]
    centers = centers[order]
    covariances = covariances[order]

    if covariance_type == 'diag':
        widths = np.sqrt(covariances)
        return centers.T, widths.T
    elif covariance_type == 'full':
        sqrt_precision_matrix = np.array(
            [inv_root_matrix(cov) for cov in covariances])
        return centers.T, sqrt_precision_matrix.T


def initialize_GaussianKernelRandom(xlims, N, covariance_type):
    xlims = np.array(xlims, dtype=np.float32)  # [d, 2]
    d = xlims.shape[0]

    # Random centers uniformly sampled within bounds
    centers = np.random.rand(d, N).astype(np.float32)
    centers = centers * (xlims[:, 1] - xlims[:, 0])[:, np.newaxis] + xlims[:, 0][:, np.newaxis]

    # Widths scaled to input range / number of centers
    widths = np.ones((d, N), dtype=np.float32)
    widths *= (xlims[:, 1] - xlims[:, 0])[:, np.newaxis] / (N / 4)

    if covariance_type == 'diag':
        initial_values = [centers, widths]
    else:  # 'full'
        sqrt_precision_matrix = np.stack(
            [np.diag(1.0 / (1e-4 + widths[:, n])).astype(np.float32) for n in range(N)],
            axis=-1
        )
        initial_values = [centers, sqrt_precision_matrix]

    return initial_values


class OuterProduct(nn.Module):
    def __init__(self, n_filters, use_single1=True, use_single2=True, use_bias=True,
                 non_negative=False, unitnorm=False, fixednorm=None,
                 symmetric=False, diagonal=False,
                 sum_axis=None, non_negative_initial=False):
        super(OuterProduct, self).__init__()

        self.n_filters = n_filters
        self.use_single1 = use_single1
        self.use_single2 = use_single2
        self.use_bias = use_bias
        self.non_negative = non_negative
        self.fixednorm = 1.0 if unitnorm else fixednorm
        self.symmetric = symmetric
        self.diagonal = diagonal
        self.sum_axis = sum_axis
        self.non_negative_initial = non_negative_initial

        # We will initialize weights in `reset_parameters()` once input shape is known
        self.built = False

    def _build_weights(self, input1_dim, input2_dim):
        self.n1 = input1_dim
        self.n2 = input2_dim

        # Calculate stddev
        if self.fixednorm is not None:
            stddev = self.fixednorm / np.sqrt(self.n1 * self.n2)
        elif self.diagonal:
            stddev = 1.0 / np.sqrt(self.n1)
        else:
            stddev = 1.0 / np.sqrt(self.n1 * self.n2)

        # Main kernel: [n1, n2, n_filters] or [n1, n_filters] if diagonal
        if self.diagonal:
            shape = (self.n1, self.n_filters)
        else:
            shape = (self.n1, self.n2, self.n_filters)

        init = torch.rand(*shape) if self.non_negative_initial else torch.randn(*shape) * stddev
        self.kernel12 = nn.Parameter(init.float())

        if self.use_single1:
            stddev1 = 1.0 / np.sqrt(self.n1)
            init1 = torch.rand(self.n1, self.n_filters) if self.non_negative_initial else torch.randn(self.n1, self.n_filters) * stddev1
            self.kernel1 = nn.Parameter(init1.float())

        if self.use_single2:
            stddev2 = 1.0 / np.sqrt(self.n2)
            init2 = self.kernel1 if self.symmetric else (
                torch.rand(self.n2, self.n_filters) if self.non_negative_initial else torch.randn(self.n2, self.n_filters) * stddev2
            )
            self.kernel2 = nn.Parameter(init2.float() if not self.symmetric else self.kernel1)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.n_filters))

        self.built = True


    def forward(self, inputs):
        first_input, second_input = inputs  # shapes: [B, ..., n1], [B, ..., n2]

        if not self.built:
            self._build_weights(first_input.size(-1), second_input.size(-1))

        # Compute outer product
        if self.diagonal:
            out = torch.matmul(first_input * second_input, self.kernel12)  # [B, ..., filters]
        else:
            # Expand for outer product
            x1 = first_input.unsqueeze(-1)  # [B, ..., n1, 1]
            x2 = second_input.unsqueeze(-2)  # [B, ..., 1, n2]
            outer = x1 * x2  # [B, ..., n1, n2]

            if self.sum_axis is not None:
                outer = outer.sum(dim=self.sum_axis)

            out = torch.tensordot(outer, self.kernel12, dims=([-2, -1], [0, 1]))  # [B, ..., filters]

        # Single branches
        if self.use_single1:
            x1_sum = first_input.sum(dim=self.sum_axis) if self.sum_axis is not None else first_input
            out += torch.matmul(x1_sum, self.kernel1)

        if self.use_single2:
            x2_sum = second_input.sum(dim=self.sum_axis) if self.sum_axis is not None else second_input
            out += torch.matmul(x2_sum, self.kernel2)

        if self.use_bias:
            out += self.bias

        return out



class MultiTanh(nn.Module):
    def __init__(self, ntanh, use_bias=True):
        super(MultiTanh, self).__init__()
        self.ntanh = ntanh
        self.use_bias = use_bias

        self.widths = None  # initialized lazily
        self.slopes = None
        self.offsets = None
        self.biases = None

    def _build(self, input_dim):
        shape = (input_dim, self.ntanh)

        self.widths = nn.Parameter(torch.ones(shape))  # Constant(1)
        self.slopes = nn.Parameter(torch.ones(shape))  # Constant(1)

        # Offset initialized between -3 and 3
        initial_offsets = np.zeros(shape, dtype=np.float32)
        if self.ntanh > 1:
            initial_offsets += (np.linspace(-3, 3, self.ntanh)[np.newaxis, :])
        self.offsets = nn.Parameter(torch.tensor(initial_offsets, dtype=torch.float32))

        if self.use_bias:
            self.biases = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        input_dim = x.size(-1)
        if self.widths is None:
            self._build(input_dim)

        x_exp = x.unsqueeze(-1)  # [B, ..., D, 1]

        widths = self.widths.view([1] * (x.dim() - 1) + list(self.widths.shape))
        slopes = self.slopes.view_as(widths)
        offsets = self.offsets.view_as(widths)

        output = slopes * torch.tanh((x_exp - offsets) / (widths + 1e-4))
        output = output.sum(dim=-1)  # sum over ntanh

        if self.use_bias:
            bias_shape = [1] * (x.dim() - 1) + [input_dim]
            output += self.biases.view(*bias_shape)

        return output


def moments_masked(x, mask, axes, keepdims=False):
    # Convert mask: [B, T] → [B, T, 1]
    mask = mask.unsqueeze(-1).float()

    # Sum along specified axes
    sum_mask = mask
    for axis in sorted(axes):
        sum_mask = sum_mask.sum(dim=axis, keepdim=True)

    sum_mask = sum_mask.clamp(min=1.0)

    mean = (x * mask).sum(dim=axes, keepdim=True) / sum_mask
    variance = ((x - mean.detach()) ** 2 * mask).sum(dim=axes, keepdim=True) / sum_mask

    if not keepdims:
        for axis in sorted(axes, reverse=True):
            mean = mean.squeeze(axis)
            variance = variance.squeeze(axis)

    return mean, variance


def normalize_batch_in_training_masking(x, mask, gamma, beta, axes, epsilon=1e-3):
    # x: [B, T, D] or [B, D]
    # mask: [B, T]
    mean, var = moments_masked(x, mask, axes, keepdims=True)

    inv = torch.rsqrt(var + epsilon)
    normed = (x - mean) * inv

    if gamma is not None:
        normed = normed * gamma
    if beta is not None:
        normed = normed + beta

    return normed, mean, var



class MaskedBatchNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-3, momentum=0.99, affine=True, track_running_stats=True):
        super(MaskedBatchNormalization, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum,
                                 affine=affine, track_running_stats=track_running_stats)

    def forward(self, x, mask=None):
        # x: [B, T, D] or [B, D]
        if mask is None or not self.training:
            # Use standard batch norm when mask not provided or in eval mode
            if x.dim() == 3:
                # [B, T, D] → [B*T, D]
                B, T, D = x.shape
                x_reshaped = x.contiguous().view(B * T, D)
                out = self.bn(x_reshaped)
                return out.view(B, T, D)
            else:
                return self.bn(x)
        else:
            # Custom masked normalization for [B, T, D]
            B, T, D = x.shape
            gamma = self.bn.weight.view(1, 1, D) if self.bn.affine else None
            beta = self.bn.bias.view(1, 1, D) if self.bn.affine else None

            out, mean, var = normalize_batch_in_training_masking(
                x, mask, gamma, beta, axes=[1], epsilon=self.bn.eps)

            # Update running stats manually
            with torch.no_grad():
                batch_mean = mean.mean(dim=0).squeeze(0)  # [D]
                batch_var = var.mean(dim=0).squeeze(0)    # [D]
                momentum = self.bn.momentum

                self.bn.running_mean.mul_(1 - momentum).add_(momentum * batch_mean)
                self.bn.running_var.mul_(1 - momentum).add_(momentum * batch_var)

            return out


class Bias(nn.Module):
    def __init__(self):
        super(Bias, self).__init__()
        self.bias = None

    def forward(self, x):
        if self.bias is None:
            self.bias = nn.Parameter(torch.zeros(x.size(-1)))

        shape = [1] * (x.dim() - 1) + [x.size(-1)]
        return x + self.bias.view(*shape)


class Slope(nn.Module):
    def __init__(self):
        super(Slope, self).__init__()
        self.slope = None

    def forward(self, x):
        if self.slope is None:
            self.slope = nn.Parameter(torch.ones(x.size(-1)))

        shape = [1] * (x.dim() - 1) + [x.size(-1)]
        return x * self.slope.view(*shape)
