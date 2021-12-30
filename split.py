import cv2
import numpy as np
import pytesseract
import uuid


import os


class Interception:
    def __init__(self):
        self.min_area = 1000    # 过滤掉太小的矩形
        self.max_area = 7000    # 手写字框的面积
        self.save_dir = "dataset/"
        self.image_dir = "image/" 
        self.handled_image = None
        self.path_list = os.listdir(self.image_dir)
        self.raw_image = None       # 读取的图片
        self.result_image = None    # 拷贝一份用于扣字
        self.kernel = np.ones((3, 3), np.uint8)
        self.contour_list = []      # 识别到的轮廓
        self.axis_contour_list = [] # 轮廓的所有信息+编号
        self.rounds = 0
        # 删除掉非图片文件
        for image_path in self.path_list:
            if os.path.splitext(image_path)[1] not in ['.jpg', '.png', '.gif', '.bmp', '.jpeg']:
                print(f"\033[1;31m!!<===\033[0m \033[1;33m{self.image_dir}\033[0m \033[1;31m文件夹有一个文件不是图片，请检查后再运行此代码===>!!\033[0m")
                # os.remove(self.image_dir + image_path)
                os._exit(1)

    def show(self, name, image):
        height, width = image.shape[:2]
        image = cv2.resize(image, (int(width/2), int(height/2)))
        cv2.imshow(name, image)
        cv2.waitKey(0)

    def readImage(self, image_name):
        # 读取图片
        image_path = self.image_dir + image_name
        self.raw_image = cv2.imread(image_path)
        h, w = self.raw_image.shape[:2]
        if w > h:
            self.raw_image = cv2.resize(self.raw_image, (1500, 1000))
        else:
            self.raw_image = cv2.resize(self.raw_image, (1000, 1500))
        self.result_image = self.raw_image.copy()

    def prehandle(self, image):
        # 双边滤波
        bilateral_filtered_image = cv2.bilateralFilter(image, 5, 175, 175)
        # 边缘提取
        edge_detected_image = cv2.Canny(bilateral_filtered_image, 90, 200)
        # 开闭操作
        # dilate_detected_image = cv2.morphologyEx(edge_detected_image, cv2.MORPH_OPEN, kernel)
        # dilate_detected_image = cv2.erode(edge_detected_image, self.kernel, iterations=2)
        dilate_detected_image = cv2.morphologyEx(edge_detected_image, cv2.MORPH_OPEN, self.kernel, iterations=1)  # 闭操作
        # self.show("test", dilate_detected_image)
        return dilate_detected_image
    
    def prehandle2(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Laplacian=cv2.Laplacian(gray,cv2.CV_8U,ksize=3)
        ret, binary = cv2.threshold(Laplacian, 127, 255, cv2.THRESH_BINARY)
        dilate_detected_image = cv2.dilate(binary, self.kernel, iterations=2)
        # dilate_detected_image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel, iterations=1)  # 闭操作
        # self.show("test", dilate_detected_image)
        return dilate_detected_image

    def do(self):
        if len(self.path_list) == 0:
            return
        for image_path in self.path_list:
            self.rounds += 1
            print("\033[1;37m 识别进度==================>{0}%  |  共{1}张图片\033[0m".format(self.rounds/len(self.path_list), len(self.path_list)))
            self.readImage(image_path)                     # 读取图片并备份
            # prehandled_image = self.prehandle(self.raw_image)  # 图像预处理
            prehandled_image = self.prehandle2(self.raw_image)
            # 找轮廓
            contours, hierarchy = cv2.findContours(prehandled_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            self.contour_list = []
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
                    cv2.rectangle(self.raw_image, (x-w-w//12, y+10), (x-w//2-w//8, y+h//2-5), (0, 0, 255), 2)
                    number = self.result_image[y+10:y+h//2-5, x-w-w//12:x-w//2-w//8]
                    gray = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)
                    # if i == 1:
                    #     cv2.imshow("name", number)
                    #     cv2.waitKey(0)
                    # i = 2
                    text = pytesseract.image_to_string(number, lang='chi_sim+eng', config="--psm 10")
                    num = ''.join(filter(str.isdigit, text))
                    if len(num) == 1:
                        num = '0'.join(num)
                    self.axis_contour_list.append([x, y, w, h, num])
                    cv2.putText(self.raw_image, num, (x-w, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
            cv2.drawContours(self.raw_image, self.contour_list, -1, (0,0,255), 2)
            # self.show('Objects Detected',self.raw_image)
            
            # 逐张保存
            for i in range(0, len(self.axis_contour_list)):
                x = self.axis_contour_list[i][0]
                y = self.axis_contour_list[i][1]
                w = self.axis_contour_list[i][2]
                h = self.axis_contour_list[i][3]
                num = self.axis_contour_list[i][4]
                image = self.result_image[y+5:y+h-5, x+5:x+w-5]
                image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
                cv2.imshow(str(i), image)
                key = cv2.waitKey(1)
                # if key == ' ':
                #     continue
                if key & 0xFF == ord('q'):
                    print('finish ------')
                    break
                else:
                    save_path = self.save_dir + '/' + str(num)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(save_path + '/' + str(num) + '-' + str(uuid.uuid4()) + ".bmp", image)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    interception = Interception()
    interception.do()