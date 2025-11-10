import kagglehub
import tensorflow as tf
from tensorflow.data import AUTOTUNE
import keras_cv
import xml.etree.ElementTree as ET
import pandas as pd
import os
from PIL import Image
import numpy as np


def download_kaggle_data():
    path = kagglehub.dataset_download("trainingdatapro/cars-video-object-tracking")
    print("Path to dataset files:", path)
    return path


def parse_annotations(annotations_path):
    tree = ET.parse(f"{annotations_path}/annotations.xml")
    root = tree.getroot()
    rows = []
    for track in root.findall('track'):
        track_id = track.get('id')
        label = track.get('label')
        for box in track.findall('box'):
            frame = int(box.get('frame'))
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            rows.append({
                'frame': frame,
                'track_id': track_id,
                'label': label,
                'xtl': xtl,
                'ytl': ytl,
                'xbr': xbr,
                'ybr': ybr
            })

    df = pd.DataFrame(rows)
    df['image'] = df['frame'].apply(lambda f: f"frame_{f:06d}.PNG")
    annotations_by_image = df.groupby('image').apply(
        lambda x: x[['xtl', 'ytl', 'xbr', 'ybr', 'track_id', 'label']].values
    ).rename('annotations').reset_index()
    annotations_by_image = annotations_by_image.iloc[:301]

    annotations_by_image['categories'] = annotations_by_image['annotations'].apply(
        lambda x: x[:, -1]
    )
    annotations_by_image['boxes'] = annotations_by_image['annotations'].apply(
        lambda x: x[:, :-2]
    )
    cat_mapping = {0: 'car', 1: 'minivan'}
    cat_mapping_r = {'car': 0, 'minivan': 1}

    return annotations_by_image, cat_mapping, cat_mapping_r


def preprocess_images_and_boxes(images, boxes, labels, image_dir, cat_mapping_r,
                                target_size=(640, 640)):
    h, w = target_size
    n = len(images)

    imgs_out = np.zeros((n, h, w, 3), dtype=np.uint8)
    boxes_out = []
    labels_out = []
    for j, (image, image_boxes, image_labels) in enumerate(zip(images, boxes, labels)):
        path = os.path.join(image_dir, image)
        with Image.open(path) as img:
            img = img.convert('RGB')
            orig_w, orig_h = img.size
            img_resized = img.resize((w, h), Image.BILINEAR)

        processed_image_boxes = np.zeros((len(image_boxes), 4), dtype=np.float32)
        for i, (xtl, ytl, xbr, ybr) in enumerate(image_boxes.tolist()):
            scale_x = w / orig_w
            scale_y = h / orig_h

            processed_image_boxes[i, 0] = xtl * scale_x
            processed_image_boxes[i, 1] = ytl * scale_y
            processed_image_boxes[i, 2] = xbr * scale_x
            processed_image_boxes[i, 3] = ybr * scale_y
        labels_out.append([cat_mapping_r[cat] for cat in image_labels])
        imgs_out[j] = img_resized
        boxes_out.append(processed_image_boxes)
    return imgs_out, boxes_out, labels_out


def load_and_pack(img, bbox_xyxy, cls):
    img = tf.cast(img, tf.float32)
    return {
        "images": img,
        "bounding_boxes": {
            "boxes": bbox_xyxy,
            "classes": tf.cast(cls, tf.int32),
        },
    }


def make_raw_dataset(images, boxes, labels):
    images_t = tf.convert_to_tensor(images, dtype=tf.float32)
    boxes_rt = tf.ragged.constant(boxes, ragged_rank=1)
    labels_rt = tf.ragged.constant(labels, ragged_rank=1, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((images_t, boxes_rt, labels_rt))
    return ds.map(
        lambda img, b, c: load_and_pack(img, b, c),
        num_parallel_calls=AUTOTUNE,
    )


def make_dataset(raw_ds, batch_size=4, shuffle_buffer=1000,
                 augment=True):
    if shuffle_buffer:
        raw_ds = raw_ds.shuffle(shuffle_buffer)
    ds = raw_ds.ragged_batch(batch_size, drop_remainder=True)
    if augment:
        augmenter = tf.keras.Sequential([
            keras_cv.layers.RandomFlip(
                mode="horizontal", bounding_box_format="xyxy"
            ),
            keras_cv.layers.JitteredResize(
                target_size=(640, 640),
                scale_factor=(0.75, 1.3),
                bounding_box_format="xyxy",
            ),
        ])
        ds = ds.map(augmenter, num_parallel_calls=AUTOTUNE)
    else:
        resize = keras_cv.layers.JitteredResize(
            target_size=(640, 640),
            scale_factor=(1.0, 1.0),
            bounding_box_format="xyxy",
        )
        ds = ds.map(resize, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda batch: (batch["images"], batch["bounding_boxes"]),
                num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def split_data(prepared_images, prepared_boxes, prepared_labels, split_ratio=0.2, ):

    n_val = int(len(prepared_images) * split_ratio)
    train_raw = make_raw_dataset(
        prepared_images[n_val:], prepared_boxes[n_val:], prepared_labels[n_val:]
    )
    val_raw = make_raw_dataset(
        prepared_images[:n_val], prepared_boxes[:n_val], prepared_labels[:n_val]
    )
    train_ds = make_dataset(train_raw, batch_size=4, shuffle_buffer=1000, augment=True)
    val_ds = make_dataset(val_raw, batch_size=4, shuffle_buffer=0, augment=False)
    return train_ds, val_ds


def run_data_preparation_pipeline():
    path = download_kaggle_data()
    annotations_by_image, cat_mapping, cat_mapping_r = parse_annotations(path)
    boxes, images, labels = annotations_by_image['boxes'].tolist(), annotations_by_image['image'].tolist(), \
    annotations_by_image['categories'].tolist()
    prepared_images, prepared_boxes, prepared_labels = preprocess_images_and_boxes(images, boxes, labels,
                                                                                   f"{path}/images", cat_mapping_r)
    train_ds, val_ds = split_data(prepared_images, prepared_boxes, prepared_labels)
    return {
        'train_ds': train_ds,
        'val_ds': val_ds,
        'prepared_images': prepared_images,
        'prepared_boxes': prepared_boxes,
        'prepared_labels': prepared_labels,
        'cat_mapping': cat_mapping,
        'cat_mapping_r': cat_mapping_r
    }
