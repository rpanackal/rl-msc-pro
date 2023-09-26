import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, n_heads=1):
        super(AttentionPooling, self).__init__()
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.n_heads = n_heads
        self.norm_factor = torch.sqrt(torch.tensor(embed_dim / n_heads, dtype=torch.float32))
    
    def forward(self, x):
        # x has shape (batch_size, seq_len, embed_dim)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        # Calculate the attention scores
        attn_scores = torch.einsum("bse,bte->bst", q, k) / self.norm_factor  # shape (batch_size, seq_len, seq_len)
        
        # Apply softmax to get the attention distribution
        attn_dist = F.softmax(attn_scores, dim=-1)  # shape (batch_size, seq_len, seq_len)
        
        # Calculate weighted sum
        pooled = torch.einsum("bst,bte->bse", attn_dist, v)  # shape (batch_size, seq_len, embed_dim)
        
        # Optionally, if you want to reduce to shape (batch_size, embed_dim)
        pooled = pooled.mean(dim=1, keepdim=True)
        
        return pooled