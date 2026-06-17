import torch

scores = torch.tensor([0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75])

probabities = torch.softmax(scores, dim=0)

print(scores)