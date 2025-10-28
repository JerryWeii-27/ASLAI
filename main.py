import os
import shutil
import random
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO


def copy_images(img_list, src_dir, dst_dir):
    """Copy a batch of images."""
    for img in img_list:
        shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))


def setup_data():
    """
    This prepares training data for YOLO.
    """

    start_time = time.perf_counter()
    SOURCE_DIR = "asl_alphabet_train"
    TRAIN_DIR = "train"
    VAL_DIR = "val"

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    for label in os.listdir(SOURCE_DIR):
        print(label)
        letter_dir = os.path.join(SOURCE_DIR, label)
        if not os.path.isdir(letter_dir):
            continue

        # This extracts all files names that ends with .jpg into a list.
        images = [
            f for f in os.listdir(letter_dir) if f.lower().endswith((".jpg"))
        ]  # I think my dataset if pure jpg.

        # Shuffle the list for whatever reason. ChatGPT says it will make training less biased.
        random.shuffle(images)

        # Split the images into 80% for training, 20% to validate training.
        split = int(0.8 * len(images))
        train_imgs = images[:split]
        val_imgs = images[split:]

        os.makedirs(os.path.join(TRAIN_DIR, label), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, label), exist_ok=True)

        max_workers = 8
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_size = len(train_imgs) // max_workers
            dst_dir = os.path.join(TRAIN_DIR, label)
            for i in range(0, len(train_imgs), chunk_size):

                cur_list = train_imgs[i : i + chunk_size]
                executor.submit(copy_images, cur_list, letter_dir, dst_dir)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_size = len(val_imgs) // max_workers
            dst_dir = os.path.join(VAL_DIR, label)
            for i in range(0, len(val_imgs), chunk_size):
                cur_list = train_imgs[i : i + chunk_size]
                executor.submit(copy_images, cur_list, letter_dir, dst_dir)

    end_time = time.perf_counter()
    delta_time = end_time - start_time
    print(f"Execution time: {delta_time:.2f} seconds")


def train_yolo():
    """
    Use YOLO to train the classification model.
    """
    model = YOLO("yolov8n-cls.pt")
    model.train(
        data="",
        epochs=30,  # Number of training cycles.
        imgsz=224,  # Image size. I used 200 here because our dataset is 200x200.
        batch=256,  # Number of images processes at once.
        lr0=0.001,  # Starting learning rate, basically means how fast weights change.
        augment=True,  # Enable random changes, like flips, rotations, and lighting to data.
        patience=5,  # Stop early if validation accuracy did not improve after 10 epochs.
    )
    metrics = model.val()
    print("\nValidation Results:")
    print(metrics)
    model.export(format="tflite")

def main():
    """
    Main
    """
    # setup_data()

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    train_yolo()


if __name__ == "__main__":
    main()
