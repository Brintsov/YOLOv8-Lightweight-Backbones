import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches


REG_MAX = 16
bins = tf.range(REG_MAX, dtype=tf.float32)
strides = tf.constant([8]*6400 + [16]*1600 + [32]*400, tf.float32)
gx = tf.constant(
    np.concatenate([np.tile(np.arange(80), 80),
                    np.tile(np.arange(40), 40),
                    np.tile(np.arange(20), 20)]).astype(np.float32))
gy = tf.constant(
    np.concatenate([np.repeat(np.arange(80), 80),
                    np.repeat(np.arange(40), 40),
                    np.repeat(np.arange(20), 20)]).astype(np.float32))


@tf.function(jit_compile=False)
def _decode_heads(raw_boxes, raw_logits,
                  score_th=0.6,
                  iou_th=0.50,
                  max_det=100):
    dfl = tf.reshape(raw_boxes, [-1, 4, REG_MAX])
    dist = tf.reduce_sum(tf.nn.softmax(dfl, -1) * bins, -1)
    cx = (gx + 0.5) * strides
    cy = (gy + 0.5) * strides
    l, t, r, b = tf.unstack(dist * strides[:, None], axis=1)

    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b
    boxes = tf.stack([x1, y1, x2, y2], axis=-1)
    boxes = tf.clip_by_value(boxes, 0., 640.)

    cls_scores = tf.sigmoid(raw_logits)
    conf = tf.reduce_max(cls_scores, axis=1)
    cls_id = tf.argmax(cls_scores, axis=1, output_type=tf.int32)

    mask = conf > score_th
    boxes_f = tf.boolean_mask(boxes, mask)
    scores_f = tf.boolean_mask(conf, mask)
    cls_f = tf.boolean_mask(cls_id, mask)

    boxes_for_nms = tf.gather(
        boxes_f,
        tf.constant([1, 0, 3, 2], tf.int32),
        axis=1
    )

    nms_id = tf.image.non_max_suppression(
        boxes_for_nms,
        scores_f,
        max_output_size=max_det,
        iou_threshold=iou_th,
        score_threshold=score_th
    )

    boxes_out = tf.gather(boxes_f, nms_id)
    scores_out = tf.gather(scores_f, nms_id)
    classes_out = tf.gather(cls_f, nms_id)

    return boxes_out, scores_out, classes_out


class YOLOv8LiteWrapper:
    def __init__(self, tflite_path,
                 device='CPU:0'):
        self.interpreter = tf.lite.Interpreter(
            model_path=tflite_path,
            num_threads=os.cpu_count()//3
        )
        self.interpreter.allocate_tensors()
        self.input_idx = self.interpreter.get_input_details()[0]['index']

        box_info = self.interpreter.get_output_details()[0]
        logit_info = self.interpreter.get_output_details()[1]
        self.box_idx = box_info['index']
        self.logit_idx = logit_info['index']
        self.box_scale = box_info['quantization_parameters']['scales']
        self.box_zp = box_info['quantization_parameters']['zero_points']
        self.log_scale = logit_info['quantization_parameters']['scales'][0]
        self.log_zp = logit_info['quantization_parameters']['zero_points'][0]
        self.device = device

    def _run(self, inp_np):
        self.interpreter.set_tensor(self.input_idx, inp_np)
        self.interpreter.invoke()
        raw_boxes, raw_logits = [
            self.interpreter.get_tensor(i['index'])[0] for i in self.interpreter.get_output_details()
        ]
        raw_boxes = (raw_boxes.astype(np.float32) -
                     self.box_zp) * self.box_scale
        raw_logits = (raw_logits.astype(np.float32) -
                      self.log_zp) * self.log_scale
        boxes, scores, classes = _decode_heads(
            tf.convert_to_tensor(raw_boxes, dtype=tf.float32),
            tf.convert_to_tensor(raw_logits, dtype=tf.float32))
        return boxes, scores, classes

    def __call__(self, x, training=False):
        boxes, scores, classes = self._run(x)
        return {"boxes": boxes,
                "scores": scores,
                "classes": classes}

    def detect_with_plot(
            self, img, class_names, score_thr=0.25
    ):
        boxes, scores, cls_ids = self(tf.cast(img, tf.float16)).values()

        if img.shape[2] == 3 and img.dtype != np.uint8:
            vis = img.astype(np.uint8)
        else:
            vis = img.copy()
        if vis.shape[2] == 3 and vis[..., ::-1].mean() < vis.mean():
            vis = vis[..., ::-1]

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(vis)
        ax.axis("off")

        for (x1, y1, x2, y2), score, cid in zip(boxes, scores, cls_ids):
            if score < score_thr:
                continue
            w, h = x2 - x1, y2 - y1
            colour = plt.cm.hsv(cid / len(class_names))
            rect = patches.Rectangle((x1, y1), w, h,
                                     linewidth=2, edgecolor=colour,
                                     facecolor="none")
            ax.add_patch(rect)
            label = f"{class_names[cid]} {score:.2f}"
            ax.text(x1, y1 - 2,
                    label,
                    fontsize=10,
                    color="white",
                    bbox=dict(facecolor=colour, edgecolor="none", pad=1.4))

        plt.tight_layout()
        plt.show()
