import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


class Quantizer:
    def __init__(self, export_path='quantized_models/', input_sig=[1, 640, 640, 3], representation_data=None):
        self.input_sig = input_sig
        self.representation_data = representation_data
        self.export_path = export_path

    def freeze_model(self, model):
        @tf.function(input_signature=[tf.TensorSpec(self.input_sig, tf.float16)])
        def inference_fn(x):
            return model(x)

        concrete_func = inference_fn.get_concrete_function()
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        return frozen_func

    def rep_ds(self):
        def ds():
            for img in self.representation_data:
                yield [np.expand_dims(img, axis=0).astype(np.float16)]
        return ds

    def quantize_model(self, model, name='model'):
        frozen_func = self.freeze_model(model)
        converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if self.representation_data is not None:
            converter.representative_dataset = self.rep_ds()
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            converter._experimental_sparse_weights = True

        tflite_quant = converter.convert()
        os.makedirs(self.export_path, exist_ok=True)
        with open(self.export_path+name, "wb") as f:
            f.write(tflite_quant)
