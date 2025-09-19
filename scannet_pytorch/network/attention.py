import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, self_attention=True, beta=True, epsilon=1e-6):
        super(AttentionLayer, self).__init__()
        self.self_attention = self_attention
        self.beta = beta
        self.epsilon = epsilon

        # These will be initialized during forward pass
        self.Lmax = None
        self.Kmax = None
        self.nfeatures_graph = None
        self.nheads = None
        self.nfeatures_output = None
        self._built = False


    def forward(self, inputs):
        if self.beta and self.self_attention:
            beta, self_attention, attn_coef, node_out, graph_weights = inputs
        elif self.beta and not self.self_attention:
            beta, attn_coef, node_out, graph_weights = inputs
        elif not self.beta and self.self_attention:
            self_attention, attn_coef, node_out, graph_weights = inputs
        else:
            attn_coef, node_out, graph_weights = inputs

        B = graph_weights.shape[0]
        self.Lmax = graph_weights.shape[1]
        self.Kmax = graph_weights.shape[2]
        self.nfeatures_graph = graph_weights.shape[-1]
        self.nheads = attn_coef.shape[-1] // self.nfeatures_graph
        self.nfeatures_output = node_out.shape[-1] // self.nheads

        if self.beta:
            beta = beta.view(B, self.Lmax, self.nfeatures_graph, self.nheads)
        if self.self_attention:
            self_attention = self_attention.view(B, self.Lmax, self.nfeatures_graph, self.nheads)

        attn_coef = attn_coef.view(B, self.Lmax, self.Kmax, self.nfeatures_graph, self.nheads)
        node_out = node_out.view(B, self.Lmax, self.Kmax, self.nfeatures_output, self.nheads)

        # Step 1: Add self-attention to diagonal attention coefficient
        if self.self_attention:
            attn_self, attn_others = torch.split(attn_coef, [1, self.Kmax - 1], dim=2)  # Split along Kmax
            attn_self = attn_self + self_attention.unsqueeze(2)  # [B, L, 1, F, H]
            attn_coef = torch.cat([attn_self, attn_others], dim=2)

        # Step 2: Multiply by inverse temperature beta
        if self.beta:
            attn_coef = attn_coef * (beta.unsqueeze(2) + self.epsilon)

        # Numerical stability trick (like softmax)
        attn_coef = attn_coef - torch.amax(attn_coef, dim=[-3, -2], keepdim=True)

        # Apply exp and weight by graph weights
        weighted = torch.exp(attn_coef) * graph_weights.unsqueeze(-1)  # [B, L, K, F, H]
        attn_final = torch.sum(weighted, dim=-2)  # sum over F → [B, L, K, H]

        # Normalize
        attn_final = attn_final / (torch.sum(torch.abs(attn_final), dim=-2, keepdim=True) + self.epsilon)
        # Weighted sum of node outputs using attention
        output = torch.sum(node_out * attn_final.unsqueeze(-2), dim=2)  # sum over K

        # Final reshape to [B, L, D_out * H]
        output = output.view(B, self.Lmax, self.nfeatures_output * self.nheads)

        return output, attn_final


