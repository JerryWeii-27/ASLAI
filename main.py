import os
import shutil
import random
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import torch
from ultralytics import YOLO


def copy_images(img_list, src_dir, dst_dir):
    """Copy a batch of images."""
    for img in img_list:
        shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))


def convert_to_grayscale(src_dir):
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".png")):
                path = os.path.join(root, f)
                img = Image.open(path).convert("L")  # L = grayscale
                img.save(path)


def setup_data():
    """
    This prepares training data for YOLO.
    """

    SOURCE_DIR = "asl_alphabet_train"
    TRAIN_DIR = "train"
    VAL_DIR = "val"

    if os.path.exists(TRAIN_DIR) or os.path.exists(VAL_DIR):
        print(f"Delete {TRAIN_DIR} and {VAL_DIR} first.")
        return

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    start_time = time.perf_counter()
    for label in os.listdir(SOURCE_DIR):
        print(label)
        letter_dir = os.path.join(SOURCE_DIR, label)
        if not os.path.isdir(letter_dir):
            continue

        # This extracts all files names that ends with .jpg into a list.
        images = [
            f for f in os.listdir(letter_dir) if f.lower().endswith((".jpg"))
        ]  # I think my dataset if pure jpg.

        # Shuffle the list.

        random.shuffle(images)
        images = images[: int(len(images) * 1)]
        # Split the images into 80% for training, 20% to validate training.
        split = int(0.8 * len(images))
        train_imgs = images[:split]
        val_imgs = images[split:]

        os.makedirs(os.path.join(TRAIN_DIR, label), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, label), exist_ok=True)

        max_workers = 16
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            chunk_size = len(train_imgs) // max_workers
            dst_dir = os.path.join(TRAIN_DIR, label)
            for i in range(0, len(train_imgs), chunk_size):
                cur_list = train_imgs[i : i + chunk_size]
                futures.append(
                    executor.submit(copy_images, cur_list, letter_dir, dst_dir)
                )
            for f in futures:
                f.result()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            chunk_size = len(val_imgs) // max_workers
            dst_dir = os.path.join(VAL_DIR, label)
            for i in range(0, len(val_imgs), chunk_size):
                cur_list = val_imgs[i : i + chunk_size]
                futures.append(
                    executor.submit(copy_images, cur_list, letter_dir, dst_dir)
                )
            for f in futures:
                f.result()
    end_time = time.perf_counter()
    delta_time = end_time - start_time
    print(f"Execution time: {delta_time:.2f} seconds")


def train_yolo():
    """
    Trains a YOLOv11n classification model on ASL data with improved regularization
    and training stability to reduce overfitting.
    """

    torch.backends.cudnn.benchmark = True

    model = YOLO("yolov8n-cls.pt")

    model.train(
        data="./",
        epochs=12,
        batch=64,
        imgsz=320,
        augment=True,  # Enable augmentations
        # Augmentation settings
        flipud=0.0,
        fliplr=0.5,  # Horizontal flips
        hsv_h=0.05,  # 5% hue variation
        hsv_s=0.7,  # Saturation
        hsv_v=0.5,  # Brightness
        degrees=20.0,  # Rotation
        translate=0.1,  # Position
        scale=0.2,  # 20% random zoom
        shear=5.0,  # Shear
        cutmix=0.1,

        patience=5,  # Early stopping
        amp=True,  # Mixed precision for speed
        optimizer="AdamW",
        lr0=5e-4,  # Lower LR, smoother convergence
        weight_decay=1e-3,  # Stronger regularization for overfaitting control
        label_smoothing=0.1,  # Softens overconfident predictions
    )

    print(model.names)
    # test_yolo()


def resume_yolo():
    """
    Resume a interrupted YOLO training.
    """
    torch.backends.cudnn.benchmark = True
    BASE_DIR = "runs/classify"
    try:
        dirs = [
            int(d[5:] or 1)
            for d in os.listdir(BASE_DIR)
            if os.path.isdir(os.path.join(BASE_DIR, d))
        ]
        last_path = os.path.join(
            BASE_DIR, "train" + str(sorted(dirs)[-1]), "weights", "last.pt"
        )
    except Exception as e:
        print(e)
        return
    model = YOLO(last_path)
    # model.train(resume=True)
    metrics = model.val()
    print("\nValidation Results:")
    print(metrics)


def main():
    """
    Main
    """
    # setup_data()

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    train_yolo()
    # resume_yolo()


if __name__ == "__main__":
    main()
