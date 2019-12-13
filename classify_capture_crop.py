"""A demo to classify Raspberry Pi camera stream."""

import argparse
import io
import time
import math

from edgetpu.classification.engine import ClassificationEngine
from edgetpu.utils import dataset_utils
import numpy as np
import picamera


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
  parser.add_argument('--label', help='File path of label file.', required=True)
  args = parser.parse_args()

  labels = dataset_utils.read_label_file(args.label)
  engine = ClassificationEngine(args.model)

  with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.framerate = 30
    camera.hflip = True
    camera.rotation = 90
    _, input_height, input_width, _ = engine.get_input_tensor_shape()

    input_size = (input_width, input_height)

    # Width is rounded up to the nearest multiple of 32,
    # height to the nearest multiple of 16.
    capture_size = (math.ceil(input_width / 32) * 32,
                    math.ceil(input_height / 16) * 16)

    camera.start_preview()
    try:
      stream = io.BytesIO()
      for _ in camera.capture_continuous(
          stream, format='rgb', use_video_port=True, resize=capture_size):
        stream.truncate()
        stream.seek(0)

        input_tensor = np.frombuffer(stream.getvalue(), dtype=np.uint8)
        if input_size != capture_size:
          # Crop to input size. Note dimension order (height, width, channels)
          input_tensor = input_tensor.reshape(
              (capture_size[1], capture_size[0], 3))[
                  0:input_height, 0:input_width, :].ravel()

        start_ms = time.time()
        results = engine.classify_with_input_tensor(input_tensor, top_k=1)
        elapsed_ms = time.time() - start_ms
        if results:
          camera.annotate_text = '%s %.2f\n%.2fms' % (
              labels[results[0][0]], results[0][1], elapsed_ms * 1000.0)
    finally:
      camera.stop_preview()


if __name__ == '__main__':
  main()
