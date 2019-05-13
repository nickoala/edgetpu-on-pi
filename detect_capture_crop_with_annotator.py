"""A demo to object-detect Raspberry Pi camera stream."""
import argparse
import io
import time
import math
import numpy as np
import picamera
from PIL import Image, ImageDraw, ImageFont
from edgetpu.detection.engine import DetectionEngine
from annotator import Annotator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
      '--label', help='File path of label file.', required=True)
    args = parser.parse_args()

    with open(args.label, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    engine = DetectionEngine(args.model)

    with picamera.PiCamera() as camera:
        preview_size = (640, 480)
        camera.resolution = preview_size
        camera.framerate = 30
        camera.hflip = True
        camera.vflip = True
        camera.rotation = 90
        _, input_width, input_height, channels = engine.get_input_tensor_shape()

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
        annotator = Annotator(camera, dimensions=annotator_size)

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
                annotator.text(
                        top_left,
                        '{} {:.2f}'.format(labels[c.label_id], c.score),
                        font=font)

            annotator.update()

        try:
            stream = io.BytesIO()
            for foo in camera.capture_continuous(stream,
                                                 format='rgb',
                                                 use_video_port=True,
                                                 resize=capture_size):
                stream.truncate()
                stream.seek(0)

                input = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                if input_size != capture_size:
                    # Crop to input size. Note dimension order (height, width, channels)
                    input = input.reshape((capture_size[1], capture_size[0], 3))[
                                0:input_height, 0:input_width, :].ravel()

                start_ms = time.time()
                results = engine.DetectWithInputTensor(input, top_k=3)
                elapsed_ms = time.time() - start_ms

                annotate(results)

                if results:
                    camera.annotate_text = "%.2fms" % (elapsed_ms*1000.0,)
        finally:
            # Maybe should make this an annotator method
            camera.remove_overlay(annotator._overlay)
            camera.stop_preview()


if __name__ == '__main__':
    main()
