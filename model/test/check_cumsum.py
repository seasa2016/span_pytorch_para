import torch

data = torch.rand(10,2).cuda()
x_len = torch.tensor([1, 2, 4, 3], dtype=torch.long).cuda()
device = data.device

max_x_len = max(x_len.cpu().tolist())
cum_len = x_len.cumsum(-1)

split_data = []
start = 0
for end in cum_len:
    split_data.append(
        torch.cat([data[start:end], torch.rand(max_x_len-end+start, 2).to(device)])
    )
    start = end

split_data = torch.stack(split_data)
print(split_data.shape)