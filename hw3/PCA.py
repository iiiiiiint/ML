import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.decomposition import PCA, IncrementalPCA
from tqdm import tqdm
import os


image_raw = imread('./deer.png')

gray_chan = False
if (image_raw.ndim) == 2:
    gray_chan = True

if not gray_chan:
    img = []
    for i in tqdm(range(image_raw.shape[-1])):
        img.append(image_raw[:, :, i])
else:
    img = image_raw

# plt.figure(figsize=(15, 5))
#
# # 子图1：显示原始图像
# plt.subplot(1, 2, 1)
# plt.imshow(image_raw)
# plt.title('Original Image')
# plt.axis('off')
#
pca = PCA()
# pca.fit(img[0])
#
# #cumulative variance
# cum_variance = np.cumsum(pca.explained_variance_ratio_) * 100
#
# #Get the number of PC whose variance > 95
# k = np.argmax(cum_variance > 95)
# kk = np.argmax(cum_variance > 98)
# kkk = np.argmax(cum_variance > 99)
# print("Number of componenets with more than 95% of variance :" + str(k))
#
# plt.subplot(1, 2, 2)
# plt.plot(np.arange(1, len(cum_variance) + 1), cum_variance)
# plt.title('Cumulative Explained Variance')
# plt.xlabel('Principal Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.axvline(x=k, color="k", linestyle="--", label=str(k))
# plt.axvline(x=kk, color="k", linestyle="--", label=str(kk))
# plt.axvline(x=kkk, color="k", linestyle="--", label=str(kkk))
# plt.axhline(y=95, color="r", linestyle="--", label="95")
# plt.axhline(y=98, color="r", linestyle="--", label="98")
# plt.axhline(y=99, color="r", linestyle="--", label="99")
# plt.legend()
#
# plt.tight_layout()
# #
# # # 显示图形
# plt.show()

var = [95, 98, 99]

components = []
for v in var:
    cum_var_list = []
    for i in range(len(img)):
        pca.fit(img[i])
        cum_variance = np.cumsum(pca.explained_variance_ratio_) * 100
        cum_var_list.append(np.argmax(cum_variance > v))
    cum_var_list = np.asarray(cum_var_list)
    components.append(cum_var_list.max())

all_num = len(cum_variance)
cnt = 0
for q, v in zip(components,  var):
    print("Optimum components for retaining {} % variance : {}".format(v, q))
for k, v in zip(components,var):
    ipca = IncrementalPCA(n_components = k)
    # plt.subplot(1, 3, cnt)
    cnt += 1
    # plt.title('Using {} (of {}) componenets for retaining {}% variance'.format(k, all_num, v))
    if not gray_chan:
        image_reconstructed = []
        for i in range(len(img)):
            x = ipca.inverse_transform(ipca.fit_transform(img[i]))
            image_reconstructed.append(x)
        im = np.stack(tuple(image_reconstructed), axis=-1)
        im = np.clip(im, 0, 1)   # jpg/jpeg: im = np.clip(im, 0, 255).astype('uint8')
    else:
        im = ipca.inverse_transform(ipca.fit_transform(img))
    plt.imshow(im)
    plt.axis('off')
    plt.savefig("{}.png".format(cnt))

# plt.tight_layout()
#
# plt.show()



