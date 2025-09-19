import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np
from functools import partial
from scannet_pytorch.preprocessing import PDB_processing
from scannet_pytorch.network import neighborhoods, embeddings, attention, utils

def l1_regularization(W, l1):
    return l1 * torch.sum(torch.abs(W))

def l12_regularization(W, l12, ndims=2, order='gaussian_feature_filter'):
    if order == 'filter_gaussian_feature':
        if ndims == 2:
            scale = W.shape[1]
            reduced = W.abs().mean(dim=1)
        elif ndims == 3:
            scale = W.shape[1] * W.shape[2]
            reduced = W.abs().mean(dim=(1, 2))
    elif order == 'gaussian_feature_filter':
        if ndims == 2:
            scale = W.shape[0]
            reduced = W.abs().mean(dim=0)
        elif ndims == 3:
            scale = W.shape[0] * W.shape[1]
            reduced = W.abs().mean(dim=(0, 1))
    else:
        raise ValueError("Unsupported order: %s" % order)
    return l12 / 2 * scale * torch.sum(reduced ** 2)

def l12group_regularization(W, l12group, ndims=2, order='gaussian_feature_filter'):
    if ndims == 2:
        return l12_regularization(W, l12group, ndims=2, order=order)
    elif ndims == 3:
        if order == 'filter_gaussian_feature':
            scale = W.shape[1] * W.shape[2]
            reduced = W.pow(2).mean(dim=-1).sqrt().mean(dim=-1)
        elif order == 'gaussian_feature_filter':
            scale = W.shape[0] * W.shape[1]
            reduced = W.pow(2).mean(dim=1).sqrt().mean(dim=0)
        else:
            raise ValueError("Unsupported order: %s" % order)
        return l12group / 2 * scale * torch.sum(reduced ** 2)

def add_nonlinearity(input_dim, activation, use_multitanh=False, ntanh=5):
    layers = []
    bn = embeddings.MaskedBatchNormalization(input_dim)
    layers.append(bn)
    if use_multitanh:
        layers.append(embeddings.MultiTanh(ntanh))
    elif activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'elu':
        layers.append(nn.ELU())
    elif activation in [None, 'linear']:
        pass
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    return nn.Sequential(*layers)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class AttributeEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = embeddings.MaskedBatchNormalization(embedding_dim)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.norm(x, mask=mask)
        return self.dropout(x)

class AttributeNormalizer(nn.Module):
    def __init__(self, attribute_dim):
        super().__init__()
        self.linear = nn.Linear(attribute_dim, attribute_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(attribute_dim))
        self.linear.weight.requires_grad = False
    def forward(self, x):
        return self.linear(x)

class ScanNet(nn.Module):
    def __init__(self,
                 nfeatures_aa=20,
                 nfeatures_atom=12,
                 nembedding_aa=16,
                 nembedding_atom=12,
                 nfilters_aa=64,
                 nfilters_atom=16,
                 nattentionheads_graph=1,
                 nfilters_graph=2,
                 activation='gelu',
                 use_attention=True,
                 dropout=0.25,
                 rotation_pooling=False,
                 with_atom=True):

        super(ScanNet, self).__init__()
#         print(f"[DEBUG] Using nfeatures_aa = {nfeatures_aa}")

        self.with_atom = with_atom
        self.activation = activation
        self.rotation_pooling = rotation_pooling
       
        # === Embed amino acid attributes ===
        if nembedding_aa is None:
            self.aa_embed = nn.Identity()
        else:
            self.aa_embed = nn.Linear(nfeatures_aa, nembedding_aa)

        # === Embed atomic attributes ===
        if with_atom:
            self.atom_embed = nn.Embedding(
                num_embeddings=nfeatures_atom + 1,
                embedding_dim=nembedding_atom,
                padding_idx=0
            )

        # === Neighborhood Embedding ===
        self.embed_aa = neighborhoods.LocalNeighborhood(
            Kmax=4,
            coordinates=['distance'],
            self_neighborhood=True,
            nrotations=1
       )


        if with_atom:
            self.embed_atom = neighborhoods.LocalNeighborhood(
                Kmax=4,
                coordinates=['distance'],
                self_neighborhood=True,
                nrotations=1
        )

       # === Output classification head ===
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "silu":
            act_fn = nn.SiLU()
        elif activation == "elu":
            act_fn = nn.ELU()
        elif activation == "leakyrelu":
            act_fn = nn.LeakyReLU(negative_slope=0.01)
        elif activation == "mish":
            act_fn = Mish()
        else:
            act_fn = nn.ReLU()  # default fallback

        # Projection layer: map 102-dim aa_features → 570-dim expected by head
        #self.proj_aa = nn.Linear(102, 570)

        self.output_head = nn.Sequential(
            nn.LazyLinear(256),
            nn.LayerNorm(256),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self,
            coord_aa,
            attr_aa,
            triplets_aa,
            indices_aa,
            coord_atom=None,
            attr_atom=None,
            triplets_atom=None,
            indices_atom=None,
            mask=None):

        # === Embed attributes ===
       B, L, F = attr_aa.shape
       embedded_attr_aa = self.aa_embed(attr_aa.view(B * L, F)).view(B, L, -1)

       if attr_atom is not None and attr_atom.ndim == 2:
           attr_atom = attr_atom.unsqueeze(-1)  # (B, La) → (B, La, 1)

       if self.with_atom and attr_atom is not None:
           B, La, Fa = attr_atom.shape
           if attr_atom.dtype != torch.long:
               attr_atom = attr_atom.long()
           embedded_attr_atom = self.atom_embed(attr_atom.view(B * La, Fa)).view(B, La, -1)
           assert embedded_attr_atom.ndim == 3, f"Expected (B, La, D), got {embedded_attr_atom.shape}"
           #print("embedded_attr_atom shape:", embedded_attr_atom.shape)
           #print("coord_atom shape:", coord_atom.shape)

           # === Atom Neighborhood Embedding ===
           atom_features_list = cp.checkpoint(self.embed_atom, [coord_atom, embedded_attr_atom])  # <- updated
           atom_features = torch.cat(atom_features_list, dim=-1)

           # === Shape sanity check ===
           if indices_atom.shape[1] != coord_atom.shape[1]:
#                print(f"[⚠️] indices_atom shape {indices_atom.shape} doesn't match coord_atom {coord_atom.shape}")
               # Try slicing indices_atom to match coord_atom
               indices_atom = indices_atom[:, :coord_atom.shape[1], :]
#                print(f"[🔧] Sliced indices_atom → {indices_atom.shape}")


           # Gather atom features back to amino acid positions
           gathered_atom_features = utils.scatter_sum(
               src=atom_features,
               index=indices_atom[:, :, 0],
               dim=1,
               dim_size=embedded_attr_aa.shape[1]
           )
           # Fix dimensionality: squeeze or reshape gathered_atom_features
           if gathered_atom_features.ndim == 4:
               gathered_atom_features = gathered_atom_features.view(gathered_atom_features.shape[0], 
                                                                    gathered_atom_features.shape[1], -1)


#            print("[DEBUG]: embedded_attr_aa:", embedded_attr_aa.shape)
#            print("[DEBUG]: gathered_atom_features:", gathered_atom_features.shape)
           embedded_attr_aa = torch.cat([embedded_attr_aa, gathered_atom_features], dim=-1)

        # === Amino Acid Neighborhood Embedding ===
       aa_features = cp.checkpoint(self.embed_aa, [coord_aa, embedded_attr_aa])  # <- updated

       if isinstance(aa_features, list):
           aa_features = torch.cat(aa_features, dim=-1)

       if aa_features.ndim == 4:
           B, L, K, D = aa_features.shape
           aa_features = aa_features.view(B, L, K * D)
           #print(f"[⚙️] Flattened aa_features → shape: {aa_features.shape}")

       # 🔧 Ensure consistent feature dimension (pad to max_dim)
       max_dim = 6032  # adjust if you know your true max feature size
       if aa_features.size(-1) < max_dim:
           pad_size = max_dim - aa_features.size(-1)
           aa_features = torch.nn.functional.pad(aa_features, (0, pad_size))

       logits = self.output_head(aa_features).squeeze(-1)  # Shape: [B, L]
       return logits
