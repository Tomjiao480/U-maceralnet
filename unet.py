import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


class Unet(object):
    _defaults = {
        "model_path": 'logs/h',  # Path to the trained model
        "num_classes": 4,  # Number of segmentation classes
        "backbone": "vgg",  # Backbone network for U-Net
        "input_shape": [512, 512],  # Input image size
        "mix_type": 0,  # Blending type for visualization
        "cuda": True,  # Enable CUDA acceleration
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # Generate color map for visualization
        if self.num_classes <= 21:
            self.colors = [
                (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
                (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
                (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
                (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                (0, 64, 128), (128, 64, 12)
            ]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # Load model
        self.generate()

        show_config(**self._defaults)

    def generate(self, onnx=False):
        """Load the U-Net model and prepare it for inference."""
        self.net = unet(num_classes=self.num_classes, backbone=self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, count=False, name_classes=None):
        """Perform segmentation on a single image."""
        image = cvtColor(image)
        old_img = copy.deepcopy(image)
        orininal_h, orininal_w = np.array(image).shape[:2]

        # Preprocess the image
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # Get prediction
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        # Count pixel occurrences of each class
        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        # Visualization of segmentation result
        seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        image = Image.fromarray(np.uint8(seg_img))

        if self.mix_type == 0:
            image = Image.blend(old_img, image, 0.7)
        return image

    def get_FPS(self, image, test_interval):
        """Calculate the frames per second (FPS) for model inference."""
        image = cvtColor(image)
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        images = torch.from_numpy(image_data)
        if self.cuda:
            images = images.cuda()

        # Measure execution time
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                     int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        return (t2 - t1) / test_interval
