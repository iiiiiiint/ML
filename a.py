import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 假设有一个二维权重张量
    weights = torch.tensor([[0.2, 0.8, 0.6],
                            [0.3, 0.9, 0.7],
                            [0.4, 0.1, 0.6]]).to(device)
    grafting_value = 0.5

    a = torch.tensor([[1.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0]]).to(device)
    b = torch.tensor([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]]).to(device)
    c = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
    # 使用 torch.where 创建掩码
    # grafting_mask = torch.where(weights > grafting_value, torch.tensor(True), torch.tensor(False))
    # weights = torch.where(weights > grafting_value, a, b).to(device)
    c = c.view(2, 8)
    # weights[grafting_mask] = a[grafting_mask]
    # weights[~grafting_mask] = b[~grafting_mask]

    # 打印结果
    # print("掩码:")
    # print(grafting_mask)
    # print("\n修改后权重")
    # print(weights)
    # print("\n数据在")
    # print(weights.device)
    print(c)
