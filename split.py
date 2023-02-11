import cv2
import numpy as np

from sklearn.cluster import KMeans

import os


class Interception:
    def __init__(self):
        self.min_area = 1000    # 过滤掉太小的矩形
        self.max_area = 7000    # 手写字框的面积
        self.save_dir = "dataset/"    # 结果保存到哪里
        self.image_dir = "image/"     # 从哪个文件夹开始识别
        self.handled_image = None
        self.path_list = os.listdir(self.image_dir)
        self.raw_image = None       # 读取的图片
        self.result_image = None    # 拷贝一份用于扣字
        self.kernel = np.ones((3, 3), np.uint8)
        self.contour_list = []      # 识别到的轮廓
        self.axis_contour_list = [] # 轮廓的所有信息+编号
        self.rounds = 0

    def show(self, name, image):
        height, width = image.shape[:2]
        image = cv2.resize(image, (int(width/2), int(height/2)))
        cv2.imshow(name, image)
        cv2.waitKey(0)
        # key = cv2.waitKey(0)
        # if key & 0xFF != ord(' '):
        #     true_number = input("请输入真实编号：")
        #     return true_number
        # else:
        #     return num

    def readImage(self, image_name):
        # 读取图片
        image_path = self.image_dir + image_name
        self.raw_image = cv2.imread(image_path)
        self.result_image = self.raw_image.copy()
    
    def prehandle(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Laplacian=cv2.Laplacian(gray,cv2.CV_8U,ksize=3)
        ret, binary = cv2.threshold(Laplacian, 180, 255, cv2.THRESH_BINARY)
        dilate_detected_image = cv2.dilate(binary, self.kernel, iterations=2)
        return dilate_detected_image

    def do(self):
        if len(self.path_list) == 0:
            return
        for page_num in self.path_list:
            if page_num == str(0):
                continue
            images = os.listdir(f'{self.image_dir}/{page_num}')
            for person_image in images:
                file_name = person_image.replace(".jpg", "")
                self.rounds += 1
                print("\033[1;37m 识别进度==================>{0}%  |  共{1}张图片\033[0m".format(self.rounds/len(self.path_list), len(self.path_list)))
                self.readImage(f'{page_num}/{person_image}')             # 读取图片并备份
                prehandled_image = self.prehandle(self.raw_image)
                # self.show("black", prehandled_image)
                # 找轮廓
                contours, hierarchy = cv2.findContours(prehandled_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                self.contour_list = []
                # axis_contour_list为所有的坐标信息，未排序
                self.axis_contour_list = []
                # i = 1
                for contour in contours:
                    approx = cv2.approxPolyDP(contour, 0.02*cv2.arcLength(contour, True), True)
                    area = cv2.contourArea(contour)
                    if area < self.min_area:
                        continue
                    if (len(approx) == 4):
                        (x, y, w, h) = cv2.boundingRect(approx)
                        if h*1.5 < w:
                            continue
                        if w*h < self.max_area:
                            continue
                        self.contour_list.append(contour) 
                        # self.show("name", self.raw_image)
                        self.axis_contour_list.append([x, y, w, h])
                        cv2.drawContours(self.result_image, self.contour_list, -1, (0,0,255), 2)
                    
                if not (len(self.axis_contour_list) == 24 or len(self.axis_contour_list) == 10):
                    wrong_path = f"wrong/{page_num}"
                    print(f"{page_num}/{person_image}页面有错误")
                    if not os.path.exists(wrong_path):
                        os.makedirs(wrong_path)
                    cv2.imwrite(f"{wrong_path}/{person_image}", self.raw_image)
                    cv2.imwrite(f"{wrong_path}/{person_image}+bw.jpg", self.result_image)
                    continue
                # lx存有x坐标，对x进行聚类
                lx = []
                for i in range(0, len(self.axis_contour_list)):
                    x = self.axis_contour_list[i][0]
                    lx.append([x])
                # 先对列进行聚类，分成3列，然后依次排序
                kmeans = KMeans(n_clusters=3, n_init="auto").fit(lx)
                xy = [[], [], []]
                for i, k_label in enumerate(kmeans.labels_):
                    if(k_label == 0):
                        xy[0].append(self.axis_contour_list[i])
                    elif(k_label == 1):
                        xy[1].append(self.axis_contour_list[i])
                    else:
                        xy[2].append(self.axis_contour_list[i])
                # 对xy进行排序，拿到排好序的列列表，每一列同样排好序
                xy = [sorted(xy[0], key=lambda x:x[1]), sorted(xy[1], key=lambda x:x[1]), sorted(xy[2], key=lambda x:x[1])]
                xy = sorted(xy, key=lambda x:x[0][0])
                # 每一页的字数
                each_num = 24
                if int(page_num) == 22:
                    each_num = 10
                # 最终的有序的一维坐标向量
                ordered_coordinate = [0]*each_num
                # 先拿到每一列
                for index_of_row, row_list in enumerate(xy):
                    # 遍历每一个,最后按行输出
                    for index_of_col, row in enumerate(row_list):
                        # 每一页都有一个偏移量（24个字）
                        num = index_of_col*3+index_of_row
                        # print(num)
                        ordered_coordinate[num] = [row, num + (int(page_num)-1)*24]

                print(f"{page_num}/{person_image}页面")
                for cordi in ordered_coordinate:
                    # print(cordi)
                    # 框出来
                    x = cordi[0][0]
                    y = cordi[0][1]
                    w = cordi[0][2]
                    h = cordi[0][3]
                    num = cordi[1]+1
                    image = self.result_image[y+5:y+h-5, x+5:x+w-5]
                    cv2.rectangle(self.raw_image, (x-w-w//14-8, y+10), (x-w//2-w//10-20, y+h//2-5), (0, 0, 255), 2)
                    cv2.putText(self.raw_image, str(num), (x-w, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    save_path = self.save_dir + '/' + str(num)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(save_path + '/' + file_name + ".png", image)
                    # self.show("sdf", self.raw_image)

if __name__ == "__main__":
    interception = Interception()
    interception.do()
    print("finish")