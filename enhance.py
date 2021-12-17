import os
import shutil
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
import numpy as np
import random

'''
    opencv数据增强
    对图片进行色彩增强、高斯噪声、水平镜像、放大、旋转、剪切
'''

# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, path_out, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    cv2.imwrite(path_out, map_coordinates(image, indices, order=1, mode='reflect').reshape(shape))

def sunset(src1, path_out):
    '''
        黄昏效果
    '''
    w = src1.shape[1]
    h = src1.shape[0]

    for xi in range(0,w):
        for xj in range(0,h):
            src1[xj,xi,0]=int(src1[xj,xi,0]*0.9)
            src1[xj,xi,1]=int(src1[xj,xi,1]*0.9)
    cv2.imwrite(path_out, src1)

def dark(src1, path_out):
    w = src1.shape[1]
    h = src1.shape[0]

    # 全部变暗
    for xi in range(0,w):
        for xj in range(0,h):
            #将像素值整体减少，设为原像素值的20%
            src1[xj,xi,0]=int(src1[xj,xi,0]*0.4)
            src1[xj,xi,1]=int(src1[xj,xi,1]*0.4)
            src1[xj,xi,2]=int(src1[xj,xi,2]*0.4)
    cv2.imwrite(path_out, src1)
    


def gaussian_noise(image, path_out_gasuss, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    cv2.imwrite(path_out_gasuss, out)


def mirror(image, path_out_mirror):
    '''
        水平镜像
    '''
    h_flip = cv2.flip(image, 1)
    cv2.imwrite(path_out_mirror, h_flip)


def resize(image, path_out_large):
    '''
        放大两倍
    '''
    height, width = image.shape[:2]
    large = cv2.resize(image, (2 * width, 2 * height))
    cv2.imwrite(path_out_large, large)


def rotate(image, path_out_rotate):
    '''
        旋转
    '''
    h,w = image.shape[:2]
    center = (w//2,h//2)
    M = cv2.getRotationMatrix2D(center, 30, 1)
    rotated_template = cv2.warpAffine(image, M, (w, h),borderMode=cv2.INTER_LINEAR, borderValue=cv2.BORDER_REPLICATE)
    cv2.imwrite(path_out_rotate, rotated_template)


def shear(image, path_out_shear):
    '''
        剪切
    '''
    h, w = image.shape[:2]
    color = image[3][12]
    random_num = random.randint(20, 40)
    if random_num % 2 == 0:
        image[20:35, 20:35] = color
    elif random_num % 3 == 0:
        image[30:45, 30:45] = color
    else:
        image[35:50, 20:35] = color
    
    cv2.imwrite(path_out_shear, image)


def main():
    image_path = './04190519-whs/'
    image_out_path = './1/'
    if not os.path.exists(image_out_path):
        os.mkdir(image_out_path)
    list = os.listdir(image_path)
    print(list)
    print("----------------------------------------")
    print("The original data path:" + image_path)
    print("The original data set size:" + str(len(list)))
    print("----------------------------------------")

    imageNameList = [
        '_dark.jpg',
        '_gaussian.jpg',
        '_angulation.jpg',
        '_mirror.jpg',
        '_rotate.jpg',
        '_shear.jpg',
        '_sunset.jpg',
        '.jpg']
    for i in range(0, len(list)):
        path = os.path.join(image_path, list[i])
        out_image_name = os.path.splitext(list[i])[0]
        for j in range(0, len(imageNameList)):
            path_out = os.path.join(
                image_out_path, str(i) + imageNameList[j])
            image = cv2.imread(path)
            if j == 0:
                dark(image, path_out)
            elif j == 1:
                gaussian_noise(image, path_out)
            elif j == 2:
                elastic_transform(image, image.shape[1] * 2, image.shape[1] * 0.07,
                                           image.shape[1] * 0.06, path_out)
            elif j == 3:
                mirror(image, path_out)
            elif j == 4:
                rotate(image, path_out)
            elif j == 5:
                shear(image, path_out)
            elif j == 6:
                sunset(image, path_out)
            else:
                shutil.copy(path, path_out)
        print(out_image_name + "success！", end='\t')
    print("----------------------------------------")
    print("The data augmention path:" + image_out_path)
    outlist = os.listdir(image_out_path)
    print("The data augmention sizes:" + str(len(outlist)))
    print("----------------------------------------")
    print("Rich sample for:" + str(len(outlist) - len(list)))


if __name__ == '__main__':
    main()

