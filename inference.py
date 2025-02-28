import time

import cv2
import numpy as np
from PIL import Image

from unet import Unet

if __name__ == "__main__":
    # Initialize U-Net model
    unet = Unet()

    # Mode selection: 'predict', 'video_camera', 'fps', 'dir_predict', 'export_onnx', 'video'
    mode = "predict"

    # Whether to count detected objects
    count = True

    # Class names for segmentation
    name_classes = ["background", "Inertinite", "Vitrinite", "Lipinite"]

    # Video settings
    video_path = 0  # Set to 0 for webcam
    video_save_path = ""  # Path to save processed video
    video_fps = 25.0  # Frames per second for saved video

    # FPS testing settings
    test_interval = 100
    fps_image_path = "img/street.jpg"

    # Directory paths for batch processing
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    # ONNX export settings
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        # Predict single images
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video_camera":
        # Capture video from webcam or specified path
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to correctly read the camera (video). Check installation and path.")

        fps = 0.0
        while True:
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(unet.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        # Measure FPS for model inference
        img = Image.open(fps_image_path)
        tact_time = unet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        # Perform batch prediction on images in a directory
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = unet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

    elif mode == "export_onnx":
        # Convert model to ONNX format
        unet.convert_to_onnx(simplify, onnx_save_path)

    elif mode == "video":
        # Process a specified video file
        while True:
            video = input("Input video filename:")
            try:
                capture = cv2.VideoCapture(video)
            except:
                print("Open Error! Try again!")
                continue
            else:
                if video_save_path != "":
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

                ref, frame = capture.read()
                if not ref:
                    raise ValueError("Failed to read video. Check the file path.")

                fps = 0.0
                while True:
                    t1 = time.time()
                    ref, frame = capture.read()
                    if not ref:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(np.uint8(frame))
                    frame = np.array(unet.detect_image(frame))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    fps = (fps + (1. / (time.time() - t1))) / 2
                    print("fps= %.2f" % (fps))
                    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                        2)

                    cv2.imshow("video", frame)
                    c = cv2.waitKey(1) & 0xff
                    if video_save_path != "":
                        out.write(frame)

                    if c == 27:
                        capture.release()
                        break
                print("Video Detection Done!")
                capture.release()
                if video_save_path != "":
                    print("Save processed video to the path :" + video_save_path)
                    out.release()
                cv2.destroyAllWindows()

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
