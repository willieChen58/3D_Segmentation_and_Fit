import torch
import torch.nn as nn

# 定义损失函数
loss_func = nn.CrossEntropyLoss(reduction="none")

# 示例预测值和目标标签
outputs = torch.tensor([[0.2, 1.0, 0.5], [1.2, 0.3, 0.8]], requires_grad=True)
targets = torch.tensor([1, 0])
print(outputs.shape)
print(targets.shape)
# 计算损失
loss1, loss2 = loss_func(outputs, targets)
print(loss1.item())
print(loss2.item())
