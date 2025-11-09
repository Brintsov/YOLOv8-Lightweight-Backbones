import tensorflow as tf
import keras_cv
from tensorflow.keras import layers, Model, Input


def channel_shuffle(x, groups=2):
    batch_size, h, w, c = tf.unstack(tf.shape(x))
    channels_per_group = c // groups
    x = tf.reshape(x, [batch_size, h, w, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    return tf.reshape(x, [batch_size, h, w, c])


class ShuffleNetV2Unit(layers.Layer):
    def __init__(self, out_channels, stride, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride
        branch_channels = out_channels // 2

        if stride == 1:
            self.branch2 = tf.keras.Sequential([
                layers.Conv2D(branch_channels, 1, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.Conv2D(branch_channels, 1, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
            ])
        else:
            self.branch1 = tf.keras.Sequential([
                layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.Conv2D(branch_channels, 1, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
            ])
            self.branch2 = tf.keras.Sequential([
                layers.Conv2D(branch_channels, 1, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.Conv2D(branch_channels, 1, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
            ])

    def call(self, x):
        if self.stride == 1:
            c = x.shape[-1] // 2
            x1, x2 = x[:, :, :, :c], x[:, :, :, c:]
            out = tf.concat([x1, self.branch2(x2)], axis=-1)
        else:
            out = tf.concat([self.branch1(x), self.branch2(x)], axis=-1)
        return channel_shuffle(out, groups=2)


def build_shufflenet_v2(
    input_shape=(640, 640, 3),
    scale_factor=0.5,
    stage_repeats=[4, 8, 4]
):
    # Standard channels from paper for scale=1.0
    base_channels = [24, 116, 232, 464, 1024]
    stage_out_channels = [int(c * scale_factor) for c in base_channels]

    inputs = Input(shape=input_shape)
    x = layers.Conv2D(stage_out_channels[0], 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # Stage 2 (P3)
    for i in range(stage_repeats[0]):
        stride = 2 if i == 0 else 1
        x = ShuffleNetV2Unit(stage_out_channels[1], stride=stride)(x)
    P3 = layers.Lambda(lambda z: z, name='P3')(x)

    # Stage 3 (P4)
    for i in range(stage_repeats[1]):
        stride = 2 if i == 0 else 1
        x = ShuffleNetV2Unit(stage_out_channels[2], stride=stride)(x)
    P4 = layers.Lambda(lambda z: z, name='P4')(x)

    # Stage 4 (P5)
    for i in range(stage_repeats[2]):
        stride = 2 if i == 0 else 1
        x = ShuffleNetV2Unit(stage_out_channels[3], stride=stride)(x)

    x = layers.Conv2D(stage_out_channels[4], 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    P5 = layers.Lambda(lambda z: z, name='P5')(x)

    model = Model(inputs, [P3, P4, P5], name=f'ShuffleNetV2_{scale_factor}x')
    model.pyramid_level_inputs = {'P3': 'P3', 'P4': 'P4', 'P5': 'P5'}

    return model


def create_0_5_shufflenet_yolov8(num_classes):
    backbonde_shuffle = build_shufflenet_v2()
    model_shuffle = keras_cv.models.YOLOV8Detector(
        num_classes=num_classes,
        bounding_box_format="xyxy",
        backbone=backbonde_shuffle,
        fpn_depth=3,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        global_clipnorm=10.0,
    )

    model_shuffle.compile(
        optimizer=optimizer,
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        jit_compile=False,
    )
    return model_shuffle
