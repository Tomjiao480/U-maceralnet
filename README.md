
# Project name
U-Maceralnet: A Deep Learning Model for Coal Maceral Images Segmentation
##  Title of manuscript
An Improved U-net Model for the Identification of Coal Macerals

# 📌 Overview
U-Maceralnet is a deep learning-based semantic segmentation model specifically designed for **identification of coal maceral images**. By integrating **Dynamic Snake Convolution** and **External Attention Mechanism**, U-Maceralnet effectively enhances the segmentation accuracy of **lathy coal macerals** while addressing the challenges in coal maceral images processing. The U-maceralnet model achieves average values of 84.01% for pixel accuracy (PA), 75.29% for intersection over union (IoU), 87.58% for precision, and 89.77% for recall.
# 🔥 Key Features

-   using **Dynamic Snake Convolution** to improve lathy structure recognition
-   using **External Attention Mechanism** to enhance feature representation
-   using **Pixel-level segmentation** to ensure detailed maceral classification
-   achving **High-accuracy segmentation** of coal maceral images

# 🚀 Installation
### Prerequisites

Ensure you have the following installed:

-   Python 3.8+
-   PyTorch 2.3.0
-   CUDA 12.1 (for GPU acceleration)
-   OpenCV, NumPy, Matplotlib, and other dependencies

### Install via Conda
[Anaconda](https://www.anaconda.com/) is strongly recommended to use a configured runtime environment
```bash
conda create -n u-maceralnet python=3.10

conda activate u-maceralnet

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

# 📂 Dataset Preparation
10 coal maceral images were placed in VOCdevkit for testing purposes
```
dataset/
└─VOC2007
    ├─ImageSets
    │  └─Segmentation
    │      ├─train.txt       # Training image list
    │      ├─val.txt         # Validation image list
    │      ├─test.txt        # Test image list
    ├─JPEGImages
    │   ├─img1.jpg
    │   ├─img2.jpg
    │   ├─...
    └─SegmentationClass
        ├─img1.png
        ├─img2.png

```
# 🛠️ Usage
## 1️⃣ Training the Model
```
python train.py --epochs  --batch_size  --lr  --gpu 
```
## 2️⃣ Evaluating the Model
```
python evaluate.py --weights --dataset 
```
## 3️⃣ Running Inference
```
python inference.py --image sample.jpg --output result.jpg
```
# 🤝 Acknowledgments
Special thanks to **coal petrography experts** for data annotation and contributions to the dataset.
# 📬 Contact
If you have any questions, feel free to reach out!
* Na Xu: xuna1011@gmail.com; xuna@cumtb.edu.cn
* Feiyang Jiao: jiaofeiyang6@gmail.com








