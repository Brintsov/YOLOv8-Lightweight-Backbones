import os
import time
import gc
import tracemalloc
import pandas as pd
import numpy as np
import psutil
import keras_cv
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


#@tf.function(jit_compile=False)
def infer(model, x, device='CPU:0'):
    with tf.device(f"/{device}"):
        return model(x, training=False)


class InferenceProfiler:
    def __init__(self, warmup_steps=10, repeats=50, use_tracemalloc=False, batch_timing=True, device='CPU:0'):
        self._warmup_steps = warmup_steps
        self._repeats = repeats
        self._use_tracemalloc = use_tracemalloc
        self._batch_timing = batch_timing
        self._device = device

    def profile(self, model, input_data, verbose=False):
        gc.collect()
        tf.keras.backend.clear_session()
        for _ in range(self._warmup_steps):
            _ = infer(model, input_data, self._device)
        if self._use_tracemalloc:
            tracemalloc.start()

        proc = psutil.Process(os.getpid())

        cpu_percent_before = proc.cpu_percent(interval=0.1)
        cpu_times_before = proc.cpu_times()
        ram_before_mb = proc.memory_info().rss / (1024 ** 2)
        latencies = []
        if self._batch_timing:
            t0 = time.perf_counter()
            for _ in range(self._repeats):
                _ = infer(model, input_data, self._device)
            total = time.perf_counter() - t0
            avg = total / self._repeats
            latencies = [avg] * self._repeats
        else:
            for _ in range(self._repeats):
                t0 = time.perf_counter()
                _ = infer(model, input_data, self._device)
                latencies.append(time.perf_counter() - t0)

        cpu_percent_after = proc.cpu_percent(interval=0.1)
        cpu_times_after = proc.cpu_times()
        ram_after_mb = proc.memory_info().rss / (1024 ** 2)
        gpu_current_mb = gpu_peak_mb = None
        try:
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            gpu_current_mb = mem_info['current'] / (1024 ** 2)
            gpu_peak_mb = mem_info['peak'] / (1024 ** 2)
        except Exception:
            pass

        arr = np.array(latencies)
        ram_delta_mb = ram_after_mb - ram_before_mb
        cpu_user_delta = cpu_times_after.user - cpu_times_before.user
        cpu_system_delta = cpu_times_after.system - cpu_times_before.system
        stats = {
            "avg_latency_sec": float(arr.mean()),
            "median_latency_sec": float(np.median(arr)),
            "p90_latency_sec": float(np.percentile(arr, 90)),
            "min_latency_sec": float(arr.min()),
            "max_latency_sec": float(arr.max()),
            "std_latency_sec": float(arr.std()),
            "fps": float(1.0 / np.median(arr)),
            "ram_delta_mb": round(ram_delta_mb, 2),
            "cpu_percent_before": cpu_percent_before,
            "cpu_percent_after": cpu_percent_after,
            "cpu_user_time_delta_s": round(cpu_user_delta, 4),
            "cpu_sys_time_delta_s": round(cpu_system_delta, 4),
            "gpu_current_mem_mb": gpu_current_mb,
            "gpu_peak_mem_mb": gpu_peak_mb,
        }
        peak_tracemalloc_mb = None
        if self._use_tracemalloc:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_tracemalloc_mb = peak / (1024 ** 2)
        if peak_tracemalloc_mb is not None:
            stats["peak_tracemalloc_mb"] = round(peak_tracemalloc_mb, 2)

        if verbose:
            print(f"Inference profiling on {self._device}:")
            for k, v in stats.items():
                print(f"  {k:24s}: {v}")
        return stats


class FlopsExtractor:

    def profile(self, model, inputs_data, verbose=False):
        inputs_data = tf.random.normal(inputs_data.shape)
        params = model.count_params()
        @tf.function
        def model_fn(x):
            return model(x)

        concrete = model_fn.get_concrete_function(inputs_data)
        flop_profile = profile(
            concrete.graph,
            options=ProfileOptionBuilder.float_operation()
        )
        total_flop_counts = flop_profile.total_float_ops
        if verbose:
            print("TOTAL FLOPS: {}".format(total_flop_counts))
        metrics = {
            "flops": total_flop_counts,
            "flops_g": total_flop_counts/1e9,
            "params_m": params/1e6,
        }
        return metrics


class COCOMetricsCalculator:
    def __init__(self, validation_data, bounding_box_format="xyxy", evaluate_freq=1, device="CPU:0"):
        self.bounding_box_format = bounding_box_format
        self.device = device
        self.validation_data = validation_data
        self.evaluate_freq = evaluate_freq
        self.metric_calculator = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1,
        )

    def profile(self, model, data, verbose=False):
        self.metric_calculator.reset_state()
        for batch in self.validation_data:
            images, y_true = batch[0], batch[1]
            y_pred = model.predict(images)
            try:
                self.metric_calculator.update_state(y_true, y_pred)
            except:
                continue
        metrics = self.metric_calculator.result()
        metrics = {
            name: float(value.numpy() if hasattr(value, "numpy") else value)
            for name, value in metrics.items()
        }
        if verbose:
            print(f"METRICS: {metrics}")
        return metrics


class COCOMetricsCalculatorLite(COCOMetricsCalculator):

    def profile(self, model, data, verbose=False):
        self.metric_calculator.reset_state()
        for batch in self.validation_data:
            images, y_trues = batch[0], batch[1]
            boxes_pred = []
            classes_pred = []
            confidence = []
            y_preds = {}
            for i, img in enumerate(images):
                y_pred = model(np.expand_dims(img, axis=0))
                boxes_pred.append(y_pred['boxes'].numpy())
                classes_pred.append(y_pred['classes'].numpy())
                confidence.append(y_pred['scores'].numpy())
            y_preds['boxes'] = tf.ragged.constant(boxes_pred, ragged_rank=1)
            y_preds['classes'] = tf.ragged.constant(classes_pred, ragged_rank=1)
            y_preds['confidence'] = tf.ragged.constant(confidence, ragged_rank=1)

            self.metric_calculator.update_state(y_trues, y_preds)
        metrics = self.metric_calculator.result()
        metrics = {
            name: float(value.numpy() if hasattr(value, "numpy") else value)
            for name, value in metrics.items()
        }
        if verbose:
            print(f"METRICS: {metrics}")
        return metrics


class ModelsProfiler:
    def __init__(self, profilers):
        self.profilers = profilers

    def profile(self, models, data):
        results = []
        for name, model in models.items():
            model_results = {}
            for profiler in self.profilers:
                print("RUNNING ", profiler, "FOR MODEL", name)
                profile_results = profiler.profile(model, data)
                model_results.update(profile_results)
            model_results['model'] = name
            results.append(model_results)
        return pd.DataFrame(results)
