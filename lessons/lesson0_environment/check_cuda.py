import torch

x = torch.tensor([1.0, 2.0, 3.0]).to("cuda")
print(x.device)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))

print("-------")
print(torch.cuda.device_count())
