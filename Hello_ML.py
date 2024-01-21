# import matplotlib.pyplot as plt
# import numpy as np
#
# y = np.array([35, 25, 25, 15])
#
# plt.pie(y,
# labels=['A', 'B', 'C', 'D'],  # 设置饼图标签
# colors=["#d5695d", "#5d8ca8", "#65a479", "#a564c9"], # 设置饼图颜色
#         )
# plt.title("ML")  # 设置标题
# plt.show()
import torch

print(torch.cuda.is_available())
