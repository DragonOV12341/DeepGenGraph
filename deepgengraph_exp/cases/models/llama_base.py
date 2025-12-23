import os
import safetensors
import safetensors.torch
import json
# import deepgengraph_exp._csrc as native_ops


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


def collect_hf_weight(hf_model_path, names=None, use_safe_tensor=True, device='cpu'):
  if use_safe_tensor:
    weight_index_fn = 'model.safetensors.index.json'
    default_weight_fn = 'model.safetensors'
  else:
    weight_index_fn = 'pytorch_model.bin.index.json'
    default_weight_fn = 'unimplemeted_error'
  
  if os.path.exists(os.path.join(hf_model_path, weight_index_fn)):
    with open(os.path.join(hf_model_path, weight_index_fn)) as f:
      mapping = json.load(f)
      mapping = mapping['weight_map']
      files = [mapping[name] for name in names] if names is not None else mapping.values()
      files = set(files)
  else:
    assert os.path.exists(os.path.join(hf_model_path, default_weight_fn)), f"{default_weight_fn}"
    files = set([default_weight_fn])

  weight_dict = {}
  if use_safe_tensor:
    print(f"using safe tensor: {files=}")
    for file_name in tqdm(files):
      file = os.path.join(hf_model_path, file_name)
      weight = safetensors.torch.load_file(file, device=device)
      for name in weight.keys():
        param = weight[name]
        weight_dict[name] = param
  else:
    print(f"using torch bin: {files=}")
    for file_name in files:
      file = os.path.join(hf_model_path, file_name)
      weight = torch.load(file, map_location=torch.device(device))
      for name in weight.keys():
        param = weight[name]
        weight_dict[name] = param

  return weight_dict

def collect_weight_dict(hf_config):
  weight_dict = collect_hf_weight(hf_config.name_or_path)
  for layer_idx in tqdm(range(hf_config.num_hidden_layers)):
    input_layernorm_weight = weight_dict.pop(f"model.layers.{layer_idx}.input_layernorm.weight")
    attn_q_proj_weight = weight_dict.pop(f"model.layers.{layer_idx}.self_attn.q_proj.weight")
    attn_k_proj_weight = weight_dict.pop(f"model.layers.{layer_idx}.self_attn.k_proj.weight")
    attn_v_proj_weight = weight_dict.pop(f"model.layers.{layer_idx}.self_attn.v_proj.weight")
    attn_qkv_proj_weight = torch.cat([attn_q_proj_weight, attn_k_proj_weight, attn_v_proj_weight], dim=0)
    attn_o_proj_weight = weight_dict.pop(f"model.layers.{layer_idx}.self_attn.o_proj.weight")
    mlp_gate_proj_weight = weight_dict.pop(f"model.layers.{layer_idx}.mlp.gate_proj.weight")
    mlp_up_proj_weight = weight_dict.pop(f"model.layers.{layer_idx}.mlp.up_proj.weight")
    mlp_gate_up_proj_weight = torch.cat([mlp_gate_proj_weight, mlp_up_proj_weight], dim=0)
    mlp_down_proj_weight = weight_dict.pop(f"model.layers.{layer_idx}.mlp.down_proj.weight")
    post_layernorm_weight = weight_dict.pop(f"model.layers.{layer_idx}.post_attention_layernorm.weight")

    weight_dict[f"layers.{layer_idx}.input_layernorm.weight"] = input_layernorm_weight
    weight_dict[f"layers.{layer_idx}.attention.qkv_proj.weight"] = attn_qkv_proj_weight
    weight_dict[f"layers.{layer_idx}.attention.out_proj.weight"] = attn_o_proj_weight
    # weight_dict[f"layers.{layer_idx}.mlp.gate_proj.weight"] = mlp_gate_proj_weight
    # weight_dict[f"layers.{layer_idx}.mlp.up_proj.weight"] = mlp_up_proj_weight
    weight_dict[f"layers.{layer_idx}.mlp.gate_up_proj.weight"] = mlp_gate_up_proj_weight
    weight_dict[f"layers.{layer_idx}.mlp.down_proj.weight"] = mlp_down_proj_weight
    weight_dict[f"layers.{layer_idx}.post_layernorm.weight"] = post_layernorm_weight
    
  embed_tokens_weight = weight_dict.pop(f"model.embed_tokens.weight")
  rms_norm_weight = weight_dict.pop(f"model.norm.weight")
  weight_dict[f"embed_tokens.weight"] = embed_tokens_weight
  weight_dict[f"rms_norm.weight"] = rms_norm_weight

  return weight_dict

  with torch.no_grad():
    for name, param in self.named_parameters():
      if 'embed_positions' not in name:
        param.copy_(weight_dict[name])
  del weight_dict


def init_cos_sin_cache(theta, dim, max_position):
  inv_freq = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=torch.float32) / dim))
  sinusoid_inp = torch.einsum("i,j -> ij", torch.arange(max_position, dtype=torch.float32), inv_freq).to(torch.float32)
  concat = torch.concat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
  concat = concat.view(1, max_position, dim).contiguous()
  return concat


def rms_norm(out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    实现 RMS Normalization。
    
    参数:
        out (torch.Tensor): 用于存储结果的输出张量 (通常是原地操作，但在 Python 中返回新张量更安全)。
        x (torch.Tensor): 输入张量。
        weight (torch.Tensor): 学习到的缩放参数 (与特征维度匹配)。
        eps (float): 防止除零的小常数。
        
    返回:
        torch.Tensor: RMS 归一化后的结果。
    """
    # 计算平方和的均值 (mean square)
    # 保持维度，以便进行广播除法
    norm_x = x.norm(p=2, dim=-1, keepdim=True)
    rms_x = norm_x * x.shape[-1]**(-0.5)
    
    # 归一化
    # out = x / (rms_x + eps) * weight  # 原始 Llama 实现中，rms_x 是 rms(x)，这里用 x.norm(p=2)/sqrt(dim)
    
    # 通常的 RMSNorm 实现
    # 计算 (x^2) 沿着最后一个维度求均值，然后开方
    rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + eps)
    
    # 归一化并应用 weight
    out = (x / rms) * weight
    
    return out

# 示例:
# B=1, S=5, D=4
# x_example = torch.randn(1, 5, 4)
# weight_example = torch.ones(4) # weight 应该与最后一个维度匹配
# out_example = torch.empty_like(x_example)
# result_rms = rms_norm(out_example, x_example, weight_example)
# print("RMSNorm 结果:", result_rms.shape)

def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    实现 SiLU (Swish) 激活函数与乘法操作 (用于 SwiGLU/Gated-SiLU 结构)。
    假设 x 的最后一个维度是结果 out (dim) 的两倍，即 x 包含 A 和 B 两部分，
    其中 A = x[..., :dim], B = x[..., dim:]
    操作: out = SiLU(A) * B
    
    参数:
        out (torch.Tensor): 用于存储结果的输出张量。
        x (torch.Tensor): 输入张量，其最后一个维度通常是 (dim * 2)。
        
    返回:
        torch.Tensor: SiLU 激活后的结果与另一半输入相乘的结果。
    """
    dim = out.shape[-1]
    
    # 假设 x 的维度是 [..., 2*dim]
    # 将 x 沿最后一个维度分成两半
    A, B = x.split(dim, dim=-1)
    
    # 应用 SiLU 激活函数
    silu_A = torch.nn.functional.silu(A)
    
    # 将 SiLU(A) 与 B 相乘
    out = silu_A * B
    
    return out

# 示例:
# num_token=5, dim=4
# x_example = torch.randn(5, 8) # 8 = 2 * 4
# out_example = torch.empty(5, 4)
# result_silu_mul = silu_and_mul(out_example, x_example)
# print("SiLU_and_Mul 结果:", result_silu_mul.shape)

import torch
import torch

import torch

def rotary_embedding_online(
    pos: torch.Tensor, 
    q: torch.Tensor, 
    k: torch.Tensor, 
    head_dim: int, 
    rope_theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    实现在线 RoPE (Rotary Positional Embedding) 旋转，修正了维度不匹配的错误。
    
    假设 q 和 k 形状为 [N, D_total]，其中 D_total 是 q/k 的总维度 (例如 4096)。
    head_dim (例如 128) 是旋转要应用的维度。
    """
    
    # 获取展平后的序列/批次长度 N 和总维度 D_total
    N, D_total = q.shape
    
    # 计算头数 (假设 D_total 可以被 head_dim 整除)
    num_heads = D_total // head_dim
    
    # 1. 重塑 Q 和 K 到 [N, Num_Heads, Head_Dim]
    # 🚨 修正点 1: 重塑张量以隔离 head_dim
    q_reshaped = q.view(N, num_heads, head_dim)
    k_reshaped = k.view(N, num_heads, head_dim)
    
    # 2. 预计算频率的倒数 (基于 head_dim=128)
    # inv_freq 形状: [head_dim // 2] = [64]
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)).to(q.device)
    
    # 3. 计算位置频率
    # pos 形状: [N] (需要匹配 q/k 的第一个维度)
    # freqs 形状: [N, head_dim // 2]
    freqs = torch.outer(pos.float(), inv_freq) 
    
    # 4. 得到 cos 和 sin 并准备广播
    # cos/sin 形状: [N, 1, head_dim // 2]
    cos = torch.cos(freqs).unsqueeze(1) 
    sin = torch.sin(freqs).unsqueeze(1)

    # 5. 辅助函数: 旋转一半的维度
    # 输入 x: [N, Num_Heads, Head_Dim]
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_half = x.shape[-1] // 2
        x0 = x[..., :x_half]
        x1 = x[..., x_half:]
        return torch.cat((-x1, x0), dim=-1) # [N, Num_Heads, Head_Dim]

    # 6. 准备用于广播的 cos/sin 嵌入
    # [N, 1, head_dim // 2] -> [N, 1, head_dim]
    cos_emb = cos.repeat_interleave(2, dim=-1)
    sin_emb = sin.repeat_interleave(2, dim=-1)
    
    # 7. 应用旋转
    # q_reshaped: [N, Num_Heads, Head_Dim]
    # cos_emb: [N, 1, Head_Dim] -> 广播到 [N, Num_Heads, Head_Dim]
    q_rotated_reshaped = (q_reshaped * cos_emb) + (rotate_half(q_reshaped) * sin_emb)
    k_rotated_reshaped = (k_reshaped * cos_emb) + (rotate_half(k_reshaped) * sin_emb)
    
    # 8. 展平回 [N, D_total] 
    q_rotated = q_rotated_reshaped.view(N, D_total)
    k_rotated = k_rotated_reshaped.view(N, D_total)
    
    return q_rotated, k_rotated
# 示例
# seq_len=5, head_dim=4
# pos_example = torch.arange(5) # [0, 1, 2, 3, 4]
# q_example = torch.randn(5, 4)
# k_example = torch.randn(5, 4)

# q_result, k_result = rotary_embedding_online(
#     pos=pos_example, 
#     q=q_example, 
#     k=k_example, 
#     head_dim=4
# )

# print(f"输入 Q 形状: {q_example.shape}")
# print(f"输出 Q 形状: {q_result.shape}")

def rotary_embedding_online_cuda(pos, q, k, head_dim, rope_theta):
  assert q.dim() == 3 and q.shape[0] == 1
  assert k.dim() == 3 and k.shape[0] == 1
  return rotary_embedding_online(pos, q.view(q.shape[1], q.shape[2]), k.view(k.shape[1], k.shape[2]), head_dim, rope_theta)

def silu_and_mul_cuda(x):
  assert x.dim() == 3 and x.shape[0] == 1
  num_token = x.shape[1]
  dim = x.shape[2] // 2
  out = torch.empty(x.shape[0], num_token, dim, dtype=x.dtype, device=x.device)
  silu_and_mul(out.view(num_token, dim), x.view(num_token, x.shape[2]))
  return out

def rms_norm_cuda(x, weight, eps):
  out = torch.empty_like(x)
  rms_norm(out, x, weight, eps)
  return out


class RmsNorm(nn.Module):
  def __init__(self, dim, eps, dtype):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
  
  def norm_torch(self, x):
    x_f32 = x.float()
    output = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
    return output.type_as(x) * self.weight
  
  def forward(self, x):
    # return self.norm_torch(x)
    return rms_norm_cuda(x, self.weight, self.eps)



class MLP(nn.Module):
  def __init__(self, hidden_size, intermediate_size, hidden_act, bias=False, dtype=None):
    super().__init__()
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.hidden_act = hidden_act
    assert self.hidden_act == 'silu'
    # self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, dtype=dtype)
    # self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, dtype=dtype)
    self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias, dtype=dtype)
    self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias, dtype=dtype)
  
  def forward(self, x):
    # gate = self.gate_proj(x)
    # gate = F.silu(gate)
    # up = self.up_proj(x)
    # out = gate * up
    # out = self.down_proj(out)
    gate_up = self.gate_up_proj(x)
    out = silu_and_mul_cuda(gate_up)
    out = self.down_proj(out)
    return out