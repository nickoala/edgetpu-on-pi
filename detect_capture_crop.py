"""A demo to object-detect Raspberry Pi camera stream."""
import argparse
import io
import time
import math
import numpy as np
import picamera
from PIL import Image, ImageDraw, ImageFont
from edgetpu.detection.engine import DetectionEngine


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

        # Font for drawing detection candidates
        font = ImageFont.truetype(
                    '/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf',
                    size=24)

        camera.start_preview()
        image = Image.new('RGB', size=preview_size, color=0)
        draw = ImageDraw.Draw(image)
        overlay = camera.add_overlay(image.tobytes(), layer=3, alpha=64, format='rgb')

        def update_overlay(candidates):
            # Clear image
            box = image.getbbox()
            if box:
                image.paste(0, box=box)

            # Get actual coordinates to draw
            def translate(relative_coord):
                return (detect_size[0] * relative_coord[0],
                        detect_size[1] * relative_coord[1])

            for c in candidates:
                top_left = translate(c.bounding_box[0])
                bottom_right = translate(c.bounding_box[1])

                draw.rectangle([top_left, bottom_right], outline=(255, 255, 255))
                draw.text(top_left, '{} {:.2f}'.format(labels[c.label_id], c.score),
                          fill=(255, 255, 255),
                          font=font)

            overlay.update(image.tobytes())

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

                update_overlay(results)

                if results:
                    camera.annotate_text = "%.2fms" % (elapsed_ms*1000.0,)
        finally:
            camera.remove_overlay(overlay)
            camera.stop_preview()


if __name__ == '__main__':
    main()
