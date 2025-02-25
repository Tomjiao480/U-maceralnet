from PIL import Image
import os


def process_images(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)


    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)


            img = Image.open(image_path).convert('RGB')
            pixels = img.load()


            width, height = img.size


            for y in range(height):
                for x in range(width):
                    if pixels[x, y] == (1, 1, 1):
                        pixels[x, y] = (255, 0, 0)
                    elif pixels[x, y] == (2, 2, 2):
                        pixels[x, y] = (0, 255, 0)
                    elif pixels[x, y] == (3, 3, 3):
                        pixels[x, y] = (0, 0, 255)


            img.save(output_path)
            print(f"Processed: {filename}")



input_folder = "miou_out/detection-results"
output_folder = "output_images"

process_images(input_folder, output_folder)
