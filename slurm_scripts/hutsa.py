import torch
import time

tensor = torch.zeros((1, 1))

# Move the tensor to the GPU
tensor = tensor.cuda()

# Print the tensor
print(tensor)

time.sleep(100000)
