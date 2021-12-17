# 手写体汉字识别数据预处理方案

## 环境安装
```bash
pip install opencv-python
pip install pytesseract
```
- pytesseract还需要去网上下载软件
- - 链接地址：https://links.jianshu.com/go?to=http%3A%2F%2Fdigi.bib.uni-mannheim.de%2Ftesseract%2Ftesseract-ocr-setup-4.00.00dev.exe
- 找到你的自定义路径，如F:\Tesseract-OCR\tesseract.exe
- 找到你安装pytesseract的路径，如E:\Anaconda3\envs\tfod\Lib\site-packages\pytesseract.py
- 点开此文件，找到tesseract_cmd将它改为你刚刚安装的tesseract.exe的路径
- 最后将中文包也放到你刚刚安装的路径即可（中文包请在utils文件夹内下载）

## 使用
完成环境安装后，运行split.py文件即可，请将你的词根表图片放置在split.py的同级目录images/下，格式建议为.png, .jpg

## 结果
结果保存在同级目录dataset/下