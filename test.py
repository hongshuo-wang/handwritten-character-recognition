import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

ll = [[307, 1336], [585, 1335], [863, 1333], [307, 1172], [585, 1170], [863, 1168], [307, 1007], [585, 1006], [863, 1004], [307, 843], [585, 842], [862, 839], [307, 678], [585, 678], [862, 675], [307, 515], [585, 512], [862, 512], [307, 349], [585, 349], [862, 346], [307, 185], [585, 183], [862, 183]]


y = [[307], [585], [863], [307], [585], [863], [307], [585], [863], [307], [585], [862], [307], [585], [862], [307], [585], [862], [307], [585], [862], [307], [585], [862]]

			 
# 调用KMeans方法, 聚类数为4个，fit()之后开始聚类
kmeans = KMeans(n_clusters=3).fit(y)
# 调用DBSCAN方法, eps为最小距离，min_samples 为一个簇中最少的个数，fit()之后开始聚类
# dbscan = DBSCAN(eps = 0.132, min_samples = 3).fit(y)

# ly存有y坐标，进行聚类
lx = []
for i in range(0, len(self.axis_contour_list)):
    x = self.axis_contour_list[i][0]
    lx.append([x])
# 先对列进行聚类，分成3列，然后依次排序
kmeans = KMeans(n_clusters=3).fit(lx)
xy = [[], [], []]
for i, k_label in enumerate(kmeans.labels_):
    if(k_label == 0):
        xy[0].append(self.axis_contour_list[i])
    elif(k_label == 1):
        xy[1].append(self.axis_contour_list[i])
    else:
        xy[2].append(self.axis_contour_list[i])
# 对xy进行排序，拿到排好序的列列表，每一列同样排好序
xy = [sorted(xy[0]), sorted(xy[1]), sorted(xy[2])]
xy = sorted(xy, key=lambda x:x[1][0])
# 先拿到每一列
for row_list in xy:
    # 遍历每一个
    for row in row_list:
        # 框出来
        x = row[0]
        y = row[1]
        w = row[2]
        h = row[3]
        cv2.rectangle(self.raw_image, (x-w-w//14-8, y+10), (x-w//2-w//10-20, y+h//2-5), (0, 0, 255), 2)
        num = '1'
        cv2.putText(self.raw_image, num, (x-w, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        self.show("sdf", self.raw_image)
exit()






