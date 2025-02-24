from PIL import Image
import os


def process_images(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开图片
            img = Image.open(image_path).convert('RGB')
            pixels = img.load()

            # 获取图片尺寸
            width, height = img.size

            # 遍历所有像素点
            for y in range(height):
                for x in range(width):
                    if pixels[x, y] == (1, 1, 1):  # 检查是否为目标颜色
                        pixels[x, y] = (255, 0, 0)  # 修改为红色
                    elif pixels[x, y] == (2, 2, 2):  # 检查是否为目标颜色
                        pixels[x, y] = (0, 255, 0)  # 修改为绿色
                    elif pixels[x, y] == (3, 3, 3):  # 检查是否为目标颜色
                        pixels[x, y] = (0, 0, 255)  # 修改为蓝色

            # 保存处理后的图片
            img.save(output_path)
            print(f"Processed: {filename}")


# 设置输入和输出文件夹
input_folder = "miou_out/detection-results"  # 替换为你的图片文件夹路径
output_folder = "output_images"  # 处理后的图片将存放在这里

# 运行处理函数
process_images(input_folder, output_folder)
