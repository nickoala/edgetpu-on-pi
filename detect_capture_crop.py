"""A demo to object-detect Raspberry Pi camera stream."""

import argparse
import io
import time
import math

from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import ImageFont
from annotator import Annotator
import numpy as np
import picamera


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
  parser.add_argument('--label', help='File path of label file.')
  args = parser.parse_args()

  labels = dataset_utils.read_label_file(args.label) if args.label else None
  engine = DetectionEngine(args.model)

  with picamera.PiCamera() as camera:
    preview_size = (640, 480)
    camera.resolution = preview_size
    camera.framerate = 30
    # camera.hflip = True
    # camera.vflip = True
    # camera.rotation = 90
    _, input_height, input_width, _ = engine.get_input_tensor_shape()

    input_size = (input_width, input_height)

    # Width is rounded up to the nearest multiple of 32,
    # height to the nearest multiple of 16.
    capture_size = (math.ceil(input_width / 32) * 32,
                    math.ceil(input_height / 16) * 16)

    # Actual detection area on preview.
    detect_size = (preview_size[0] * input_size[0] / capture_size[0],
                   preview_size[1] * input_size[1] / capture_size[1])

    # Make annotator smaller for efficiency.
    annotator_factor = 0.5
    annotator_size = (int(preview_size[0] * annotator_factor),
                      int(preview_size[1] * annotator_factor))

    # Font for drawing detection candidates
    font = ImageFont.truetype(
                '/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf',
                size=12)

    camera.start_preview()
    annotator = Annotator(camera,
                          dimensions=annotator_size,
                          default_color=(255, 255, 255, 64))

    def annotate(candidates):
      annotator.clear()

      # Get actual coordinates to draw
      def translate(relative_coord):
        return (detect_size[0] * relative_coord[0] * annotator_factor,
                detect_size[1] * relative_coord[1] * annotator_factor)

      for c in candidates:
        top_left = translate(c.bounding_box[0])
        bottom_right = translate(c.bounding_box[1])

        annotator.bounding_box(top_left + bottom_right)

        text = '{} {:.2f}'.format(labels[c.label_id], c.score) \
                if labels else '{:.2f}'.format(c.score)

        annotator.text(top_left, text, font=font)

      annotator.update()

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
        results = engine.detect_with_input_tensor(input_tensor, top_k=3)
        elapsed_ms = time.time() - start_ms

        annotate(results)

        camera.annotate_text = '{:.2f}ms'.format(elapsed_ms * 1000.0)

    finally:
      # Maybe should make this an annotator method
      camera.remove_overlay(annotator._overlay)
      camera.stop_preview()


if __name__ == '__main__':
  main()
