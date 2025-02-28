import os
from PIL import Image
from tqdm import tqdm

from unet import Unet
from utils.utils_metrics import compute_mIoU, show_results

if __name__ == "__main__":
    # Mode selection:
    # 0 - Perform both prediction and mIoU calculation
    # 1 - Only perform prediction
    # 2 - Only perform mIoU calculation
    miou_mode = 2

    # Number of segmentation classes (including background)
    num_classes = 4

    # Class names corresponding to segmentation labels
    name_classes = ["background", "Inertinite", "Vitrinite", "Lipinite"]

    # Path to dataset directory
    VOCdevkit_path = 'VOCdevkit'

    # Read validation image IDs from the text file
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()

    # Ground truth label directory
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")

    # Output directory for mIoU results
    miou_out_path = "miou_out"

    # Directory for storing model predictions
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        # Create the directory for predictions if it does not exist
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        # Initialize the U-Net model
        unet = Unet()
        print("Load model done.")

        print("Get predict result.")
        # Generate segmentation predictions for each validation image
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = unet.get_miou_png(image)  # Get predicted segmentation mask
            image.save(os.path.join(pred_dir, image_id + ".png"))  # Save the predicted mask
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get mIoU.")
        # Compute mean Intersection over Union (mIoU) and related metrics
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get mIoU done.")

        # Display and save the computed evaluation results
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
