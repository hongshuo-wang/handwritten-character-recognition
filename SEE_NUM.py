from PIL import Image
import pytesseract
import argparse
import cv2
import os

# 构造并解析参数
ap = argparse.ArgumentParser()

ap.add_argument("-p", "--preprocess", type=str, default="thresh",
    help="要执行的预处理类型,有thresh 或者 blur两种")
args = vars(ap.parse_args())

# 加载图像并将其转换为灰度图
image = cv2.imread('3.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检查是否应该应用阈值来预处理,阈值化处理便于从背景中分割出前景
if args["preprocess"] == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# 检查是否应进行中值模糊处理以消除模糊,应用中值滤波有助于减少噪声
elif args["preprocess"] == "blur":
    gray = cv2.medianBlur(gray, 3)

# 将灰度图像作为临时文件写入磁盘
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# 将图像加载为PIL/Pillow图像，应用OCR，然后删除

text = pytesseract.image_to_string(Image.open(filename), config="digits")
os.remove(filename)

print(text)

# 现实输出图像
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)



