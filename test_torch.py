import torch

print('torch.__version__ =', torch.__version__)
print('torch.backends.mps.is_available() =', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)
print('torch.cuda.is_available() =', torch.cuda.is_available())
print('device_count =', torch.cuda.device_count())
print('default device =', 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

# quick tensor test
x = torch.rand(3, 3)
print('x device =', x.device)
print('x@x =\n', x @ x)
