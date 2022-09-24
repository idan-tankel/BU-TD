import torch
t = torch.tensor([[20, 29,  9,  7, 31, 35], [9, 23, 30,  5,  2, 31], [6, 27, 19,  5, 30,  7], [29,  4, 36,  2, 20, 27], [
                 32,  2, 38, 10, 35,  1], [27, 40, 13, 10, 20,  3], [6, 20, 30,  7, 40, 39]], dtype=torch.float)
kernel = torch.zeros (4, 1, 3, 3)
kernel[0, 0, 1, 0] = 1.0   # right
kernel[1, 0, 1, 2] = 1.0   # left
kernel[2, 0, 0, 1] = 1.0   # up
kernel[3, 0, 2, 1] = 1.0   # down
border_value = 47
tpad = torch.nn.functional.pad (t, (1, 1, 1, 1), mode = 'constant', value = border_value)
tsur = torch.nn.functional.conv2d (tpad.unsqueeze (0), kernel).permute (1, 2, 0)
print (tsur.shape)
print(t.shape)
