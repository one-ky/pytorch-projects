# import torch

# # Check CUDA details
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")
# print(f"Device name: {torch.cuda.get_device_name(0)}")

# # Create tensors on GPU to test speed
# a = torch.randn(1000, 1000, device='cuda')
# b = torch.randn(1000, 1000, device='cuda')

# # Test basic matrix multiplication using tensor cores
# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)

# start.record()
# c = torch.matmul(a, b)
# end.record()

# torch.cuda.synchronize()
# print(f"Matrix multiplication time: {start.elapsed_time(end):.2f} ms")

import torch
from torch.cuda.amp import autocast

def run_benchmark():
    # Enable TF32 (Tensor Float 32) - this should be enabled by default but let's make sure
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
    
    # Test sizes
    size = 4000
    
    # Create tensors
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    # Warmup GPU
    for _ in range(10):
        torch.matmul(a[:100, :100], b[:100, :100])
    
    # Regular precision test
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    c = torch.matmul(a, b)
    end.record()
    
    torch.cuda.synchronize()
    regular_time = start.elapsed_time(end)
    print(f"\nRegular precision multiplication time: {regular_time:.2f} ms")
    
    # Test with automatic mixed precision
    with autocast(device_type='cuda'):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        c = torch.matmul(a, b)
        end.record()
        
        torch.cuda.synchronize()
        amp_time = start.elapsed_time(end)
        print(f"Mixed precision multiplication time: {amp_time:.2f} ms")
        print(f"Speed improvement with AMP: {(regular_time - amp_time) / regular_time * 100:.1f}%")

run_benchmark()