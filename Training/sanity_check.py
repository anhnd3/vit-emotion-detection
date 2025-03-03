import torch
import torch.nn.functional as F

# Create dummy tensors of shape (batch, heads, seq_len, head_dim)
q = torch.randn(2, 4, 16, 64, device="xpu")
k = torch.randn(2, 4, 16, 64, device="xpu")
v = torch.randn(2, 4, 16, 64, device="xpu")
try:
    output = F.scaled_dot_product_attention(q, k, v)
    print("Test passed, output shape:", output.shape)
except Exception as e:
    print("Test failed:", e)

