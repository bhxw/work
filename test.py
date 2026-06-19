import torch
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'显卡型号: {torch.cuda.get_device_name(0)}')