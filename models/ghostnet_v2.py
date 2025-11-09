import math
import keras_cv
from tensorflow import keras
from tensorflow.keras import layers


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class GhostModule(layers.Layer):
    def __init__(
        self,
        out_channels,
        ratio=2,
        kernel_size=1,
        dw_kernel=3,
        strides=1,
        use_relu=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        init_channels = int(math.ceil(out_channels / ratio))
        self.conv1 = layers.Conv2D(
            init_channels, kernel_size, strides=strides,
            padding='same', use_bias=False
        )
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation('relu' if use_relu else 'hard_swish')
        self.dw = layers.DepthwiseConv2D(
            dw_kernel, strides=1, depth_multiplier=ratio-1,
            padding='same', use_bias=False
        )
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation('relu' if use_relu else 'hard_swish')
        self.concat = layers.Concatenate(axis=-1)
        self.slice = layers.Lambda(
            lambda x: x[..., :self.out_channels]
        )

    def call(self, inputs, training=None):
        x1 = self.conv1(inputs)
        x1 = self.bn1(x1, training=training)
        x1 = self.act1(x1)

        x2 = self.dw(x1)
        x2 = self.bn2(x2, training=training)
        x2 = self.act2(x2)

        x = self.concat([x1, x2])
        return self.slice(x)

class GhostBottleneck(layers.Layer):
    def __init__(
        self,
        out_channels,
        hidden_dim,
        dw_kernel=3,
        strides=1,
        use_se=False,
        use_dfc=False,
        dfc_pool_size=2,
        dfc_kernel=7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.dw_kernel = dw_kernel
        self.strides = strides
        self.use_se = use_se
        self.use_dfc = use_dfc

        self.ghost1 = GhostModule(hidden_dim, use_relu=True)

        if strides > 1:
            self.down_dw = layers.DepthwiseConv2D(
                dw_kernel, strides=strides,
                padding='same', use_bias=False
            )
            self.down_bn = layers.BatchNormalization()
        else:
            self.down_dw = None

        if use_dfc:
            self.dfc_pool = layers.AveragePooling2D(pool_size=dfc_pool_size)
            self.dfc_h = layers.DepthwiseConv2D(
                (1, dfc_kernel), padding='same', use_bias=False
            )
            self.dfc_v = layers.DepthwiseConv2D(
                (dfc_kernel, 1), padding='same', use_bias=False
            )
            self.dfc_up = layers.UpSampling2D(size=dfc_pool_size, interpolation='bilinear')
            self.dfc_sigmoid = layers.Activation('sigmoid')

        if use_se:
            self.se_gap = layers.GlobalAveragePooling2D(keepdims=True)
            self.se_reshape = layers.Reshape((1, 1, hidden_dim))
            self.se_conv1 = layers.Conv2D(hidden_dim // 4, 1, padding='same', use_bias=True)
            self.se_act = layers.Activation('relu')
            self.se_conv2 = layers.Conv2D(hidden_dim, 1, padding='same', use_bias=True)
            self.se_gate = layers.Activation('hard_sigmoid')
        else:
            self.se_gap = None

        self.ghost2 = GhostModule(out_channels, use_relu=False)

        self.short_dw = None
        self.short_bn1 = None
        self.short_pw = None
        self.short_bn2 = None

        self.add = layers.Add()

    def build(self, input_shape):
        in_channels = input_shape[-1]
        if self.strides > 1 or in_channels != self.out_channels:
            self.short_dw = layers.DepthwiseConv2D(
                self.dw_kernel, strides=self.strides,
                padding='same', use_bias=False
            )
            self.short_bn1 = layers.BatchNormalization()
            self.short_pw = layers.Conv2D(self.out_channels, 1, padding='same', use_bias=False)
            self.short_bn2 = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.ghost1(inputs, training=training)
        if self.down_dw:
            x = self.down_dw(x)
            x = self.down_bn(x, training=training)
        if self.use_dfc:
            dfc = self.dfc_pool(x)
            dfc = self.dfc_h(dfc)
            dfc = self.dfc_v(dfc)
            dfc = self.dfc_up(dfc)
            x = x * self.dfc_sigmoid(dfc)
        if self.se_gap:
            se = self.se_gap(x)
            se = self.se_reshape(se)
            se = self.se_conv1(se)
            se = self.se_act(se)
            se = self.se_conv2(se)
            se = self.se_gate(se)
            x = x * se
        x = self.ghost2(x, training=training)
        if self.short_dw:
            sc = self.short_dw(inputs)
            sc = self.short_bn1(sc, training=training)
            sc = self.short_pw(sc)
            sc = self.short_bn2(sc, training=training)
        else:
            sc = inputs
        return self.add([x, sc])


def build_ghostnet_backbone(
    input_shape=(640, 640, 3),
    width_multiplier=0.5,
    depth_multiplier=0.5,
    widths=[16, 32, 64, 96, 122, 144],
    repeats=[1, 1, 2, 2, 2, 1],
    strides=[1, 2, 2, 2, 1, 2],
    final_channels=960,
    use_se_stages=(2, 4),
    use_dfc_stages=(1, 2, 3)
):
    # Scale widths and repeats
    widths = [make_divisible(w * width_multiplier) for w in widths]
    repeats = [int(math.ceil(r * depth_multiplier)) for r in repeats]
    final_channels = make_divisible(final_channels * width_multiplier)

    inp = layers.Input(shape=input_shape, name='input')
    x = layers.Conv2D(widths[0], 3, strides=2, padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('hard_swish')(x)

    P3 = P4 = P5 = None
    total_stride = 2

    for stage_idx, (w, r, s) in enumerate(zip(widths, repeats, strides)):
        for i in range(r):
            stride = s if i == 0 else 1
            x = GhostBottleneck(
                out_channels=w,
                hidden_dim=w,
                dw_kernel=3,
                strides=stride,
                use_se=(stage_idx in use_se_stages),
                use_dfc=(stage_idx in use_dfc_stages),
                name=f'ghostb_stage{stage_idx}_unit{i}'
            )(x)
            total_stride *= stride

        if total_stride == 8 and P3 is None:
            P3 = layers.Lambda(lambda y: y, name='P3')(x)
        if total_stride == 16 and P4 is None:
            P4 = layers.Lambda(lambda y: y, name='P4')(x)
        if total_stride == 32 and P5 is None:
            P5 = layers.Lambda(lambda y: y, name='P5')(x)

    x = layers.Conv2D(final_channels, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('hard_swish')(x)
    if P5 is None:
        P5 = x

    model = keras.Model(inp, [P3, P4, P5], name=f'GhostNetV2_x{width_multiplier}')
    model.pyramid_level_inputs = {'P3': 'P3', 'P4': 'P4', 'P5': 'P5'}
    return model


def create_0_5_ghostnet_yolov8(num_classes, **kwargs):
    backbone = build_ghostnet_backbone(**kwargs)
    ghost_model = keras_cv.models.YOLOV8Detector(
        num_classes=num_classes,
        bounding_box_format='xyxy',
        backbone=backbone,
        fpn_depth=3
    )
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, global_clipnorm=1.0)
    ghost_model.compile(
        optimizer=optimizer,
        classification_loss='binary_crossentropy',
        box_loss='ciou',
        jit_compile=False
    )
    return ghost_model
