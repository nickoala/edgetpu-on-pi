# Raspberry Pi Supplement to Coral Edge TPU Demo

Additional sample code to run **image classification** and **object detection**
on a [Coral USB accelerator](https://coral.ai/products/accelerator) from a
Raspberry Pi, aiming to fill gaps in the [official
demos](https://github.com/google-coral/edgetpu/tree/master/examples):

- I would like to see *object detection working on a video stream* where
  rectangles are drawn on the preview to indicate object locations.

- Because Pi Camera's frame *width* is restricted to multiples of 32 and
  *height* to multiples of 16, it cannot match some models' required input size.
  For example, if you tell Pi Camera to resize to Inception V3's required
  299x299, it actually rounds up to 320x304. The captured image has to be
  cropped before passed to the Edge TPU.

Resulting files are:

1. `classify_capture.py`: Original official demo, kept for reference.

2. `classify_capture_crop.py`: This preemptively checks the model input size
against Pi Camera's capable frame size. The image is cropped, if necessary,
before passed to Edge TPU.

3. `detect_capture_crop.py`: Use the `Annotator` class borrowed from AIY Vision
Kit to draw bounding boxes around detected objects. It also crops images if
camera's frame size cannot match model input size.
