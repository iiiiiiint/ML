import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

# 读取图像
image = io.imread('deer.png')

# 获取图像的形状和像素值
height, width, channels = image.shape
pixels = image.reshape((height * width, channels))

# 设置聚类数量
k = 16

# 使用K均值聚类
kmeans = KMeans(n_clusters=k)
kmeans.fit(pixels)

# 获取簇中心和标签
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# 将图像像素替换为簇中心颜色
compressed_image = centers[labels].reshape((height, width, channels))

# 显示原始图像和压缩后的图像
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title('Original Image')

# plt.subplot(1, 2, 2)
plt.imshow(compressed_image.astype(np.uint8))
plt.title('Compressed Image')

# plt.show()
plt.savefig("new.png")
