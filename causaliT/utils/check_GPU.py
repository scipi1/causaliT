
import sys
print(sys.path)
sys.path.remove('C:\\Users\\ScipioneFrancesco\\AppData\\Roaming\\Python\\Python311\\site-packages')
print(sys.path)
import torch

print(f'Pytorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
print(torch.cuda.is_available())
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
print(f'Torch version: {torch.__version__}')