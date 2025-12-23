import json
import click
import torch
import numpy as np
import random
import os

from deepgengraph_exp.cases.kernels import KERNEL_ZOO
from deepgengraph_exp.cases.models import MODEL_ZOO

from deepgengraph_exp.utils import perf

from compile import compile

# from transformers import AutoConfig, AutoTokenizer


import torch
import torch.nn as nn
import json
import os
from typing import Optional

class SimpleConfig:
    def __init__(self, 
                 vocab_size=32000,
                 hidden_size=4096,
                 num_hidden_layers=32,
                 max_position_embeddings=2048,
                 rms_norm_eps=1e-6,
                 torch_dtype=torch.float16,
                 rope_theta=10000.0):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.torch_dtype = torch_dtype
        self.rope_theta = rope_theta
        self.num_attention_heads = 32
        self.num_key_value_heads = 32
        self.cache_budget = 512
        self.rotary_base = rope_theta
        self.intermediate_size = 11008
        self.hidden_act = 'silu'
        

def llm_setup_random(seqlen=512, layer_num=32, batch_size=1, vocab_size=32000, hidden_size=4096):
    """不依赖外部文件的随机初始化版本"""
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    # 创建配置对象
    hf_config = SimpleConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=layer_num,
        max_position_embeddings=seqlen,
        torch_dtype=torch.float16,
        rope_theta=10000.0
    )
    
    # 设置缓存预算和其他参数
    cache_budget = 512
    assert cache_budget < seqlen
    hf_config.cache_budget = cache_budget
    hf_config.roco_recent = 256
    hf_config.tau = 1.5
    
    # 创建corm_mask
    corm_mask = torch.ones(seqlen, seqlen, dtype=torch.float32, device=device)
    for i in range(seqlen):
        corm_mask[i] /= i + 1
    hf_config.corm_mask = corm_mask
    
    # 设置rotary_base
    hf_config.rotary_base = getattr(hf_config, 'rope_theta', 10000.0)
    print(f"{hf_config.num_hidden_layers=}")
    
    # 确保最大位置嵌入足够
    assert hf_config.max_position_embeddings >= seqlen
    hf_config.max_position_embeddings = seqlen
    
    # 生成随机token_ids（替代从文件读取）
    # 使用均匀分布生成在 [0, vocab_size) 范围内的随机token
    token_ids = torch.randint(0, vocab_size, (batch_size, seqlen), dtype=torch.int64, device=device)
    token_ids = token_ids.contiguous()
    
    print(f"{token_ids.shape=}")
    print(f"{token_ids.grad=}")
    
    return hf_config, token_ids

# 更灵活的版本，可以接受自定义配置
def llm_setup_custom(config: Optional[SimpleConfig] = None, 
                    seqlen=4096, 
                    batch_size=1,
                    **kwargs):
    """支持自定义配置的版本"""
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    # 如果没有提供配置，使用默认配置
    if config is None:
        config = SimpleConfig(
            vocab_size=kwargs.get('vocab_size', 32000),
            hidden_size=kwargs.get('hidden_size', 4096),
            num_hidden_layers=kwargs.get('layer_num', 32),
            max_position_embeddings=seqlen,
            torch_dtype=kwargs.get('torch_dtype', torch.float16),
            rope_theta=kwargs.get('rope_theta', 10000.0)
        )
    
    # 设置缓存相关参数
    cache_budget = kwargs.get('cache_budget', 512)
    assert cache_budget < seqlen
    config.cache_budget = cache_budget
    config.roco_recent = kwargs.get('roco_recent', 256)
    config.tau = kwargs.get('tau', 1.5)
    
    # 创建corm_mask
    corm_mask = torch.ones(seqlen, seqlen, dtype=torch.float32, device=device)
    for i in range(seqlen):
        corm_mask[i] /= i + 1
    config.corm_mask = corm_mask
    
    config.rotary_base = getattr(config, 'rope_theta', 10000.0)
    print(f"{config.num_hidden_layers=}")
    
    # 确保最大位置嵌入足够
    assert config.max_position_embeddings >= seqlen
    config.max_position_embeddings = seqlen
    
    # 生成随机token_ids
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seqlen), 
                             dtype=torch.int64, device=device).contiguous()
    
    print(f"{token_ids.shape=}")
    print(f"Device: {device}")
    
    return config, token_ids

# 创建模拟tokenizer的简单类（如果需要）
class MockTokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
    
    def __call__(self, text):
        # 模拟tokenizer行为，返回随机token
        seqlen = len(text.split()) if isinstance(text, str) else 100
        return {'input_ids': torch.randint(0, self.vocab_size, (seqlen,)).tolist()}


# def llm_setup(weight_path, seqlen, layer_num):
#   device = torch.cuda.current_device()

#   hf_config = AutoConfig.from_pretrained(weight_path)
#   cache_budget = 512
#   assert cache_budget < seqlen
#   hf_config.cache_budget = cache_budget
#   hf_config.roco_recent = 256
#   hf_config.tau = 1.5
#   corm_mask = torch.ones(seqlen, seqlen, dtype=torch.float32, device=device)
#   for i in range(seqlen):
#     corm_mask[i] /= i + 1
#   hf_config.corm_mask = corm_mask


#   if layer_num is not None:
#     hf_config.num_hidden_layers = layer_num
#   hf_config.rotary_base = getattr(hf_config, 'rope_theta', 10000.0)
#   print(f"{hf_config.num_hidden_layers=}")

#   assert hf_config.max_position_embeddings >= seqlen
#   hf_config.max_position_embeddings = seqlen

#   batch_size = 1
#   data_path = os.path.dirname(os.path.abspath(__file__)) + "/vcsum.jsonl"
#   print(f"{data_path=}")
#   with open(data_path, "r", encoding='utf-8') as f:
#     data = json.loads(f.readline())
#     data = data['context']
#   token_ids = AutoTokenizer.from_pretrained(weight_path)(data).input_ids[:seqlen]
#   assert len(token_ids) == seqlen
#   token_ids = torch.tensor(token_ids, dtype=torch.int64, device=device).reshape(batch_size, seqlen).contiguous()
#   print(f"{token_ids.shape=}")
#   print(f"{token_ids.grad=}")

#   return hf_config, token_ids


@click.command()
@click.option('--model', '-m', default='attn', help='Model name')
@click.option('--system', '-s', default='torch', help='System name')
@click.option('--seqlen', type=int, default=4096, help='seqlen')
@click.option('--layer_num', type=int, default=32, help='layer_num')
@click.option('--platform', '-p', default='yes', help='platform(yes, qiyuan, fuse0)')
def main(model, system, seqlen, layer_num, platform):
  print(f"{model=} {system=} {seqlen=} {layer_num=}")
  assert model in KERNEL_ZOO, f"model {model} not found in KERNEL_ZOO {KERNEL_ZOO.keys}"
  seed = 0
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

  kernel_cls = KERNEL_ZOO[model]
  model_cls = MODEL_ZOO[model]

  kernel = kernel_cls().eval().cuda()
  specs = kernel.prepare(q_len=seqlen, kv_len=seqlen)
  input_names = list(specs['input'].keys())
  inputs = [specs['input'][name] for name in input_names]
  output_names = specs['output']

  print(f"{input_names=}")
  print(f"{output_names=}")

  assert system in ['torch', 'tensorrt', 'tvm', 'xla', 'korch', 'einnet', 'our', 'dynamo']
  kernel_f = compile(
    model=kernel,
    input_names=input_names,
    inputs=inputs,
    output_names=output_names,
    system=system,
  )

  # weight_zoo_path = os.path.dirname(os.path.abspath(__file__)) + "/weight_zoo.json"
  # print(f"{weight_zoo_path=}")
  # with open(weight_zoo_path, "r") as f:
  #   weight_zoo = json.load(f)

  # weight_path = weight_zoo[platform]
  hf_config, token_ids = llm_setup_random(seqlen, layer_num)

  # build model
  model = model_cls(
    hf_config=hf_config,
    attn_f=kernel_f,
  )
  for param in model.parameters():
    param.requires_grad = False
  model = model.eval().cuda()

  run = 50
  warmup = 50
  perf(
    label=system,
    f=model,
    args=(token_ids,),
    run=run,
    warmup=warmup,
    profile=True,
  )

if __name__ == '__main__':
  main()