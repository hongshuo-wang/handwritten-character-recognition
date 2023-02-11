import fitz
import os


data_path = "data"


def pdf2img(pdf_path, zoom_x, zoom_y, people_id):
    doc = fitz.open(pdf_path)  # 打开文档
    print(len(doc))
    if len(doc) < 23:
        print(f"{pdf_path}共{len(doc)}页，请检查")
    for i, page in enumerate(doc):  # 遍历页面
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom_x, zoom_y), colorspace='rgb')  # 将页面渲染为图片
        if not os.path.exists(f'image/{page.number}'):
            os.makedirs(f'image/{page.number}')
        pix.writePNG(f'image/{page.number}/{people_id}.jpg')  # 将图像存储为PNG格式
    doc.close()  # 关闭文档



path_list = os.listdir(data_path)
for data in path_list:
    pdf2img(f'{data_path}/{data}/{data}.pdf', 2, 2, data)

