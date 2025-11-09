import keras_cv
import tensorflow as tf


def create_xs_csp_darknet_yolov8(num_classes):
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_xs_backbone",
    )
    model = keras_cv.models.YOLOV8Detector(
        num_classes=num_classes,
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=3,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        global_clipnorm=10.0,
    )

    model.compile(
        optimizer=optimizer,
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        jit_compile=False,
    )
    return model