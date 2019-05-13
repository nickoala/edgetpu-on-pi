# Raspberry Pi Supplement to Coral Edge TPU Demo

I have recently tried out the [Coral USB
Accelerator](https://coral.withgoogle.com/tutorials/accelerator/). The software
and docs are generally good. The Python library even includes a demo showing how
to capture continuously using a Pi Camera and pass the data to the Edge TPU for
real-time classification.

In terms of integrating with Raspberry Pi, though, there are still gaps to fill.
First, I would like an example for *object detection* where rectangles are drawn
on the preview to indicate object locations. Second, because Pi Camera's frame
*width* is restricted to multiples of 32 and *height* to multiples of 16, it
simply cannot match some models' required input size. For example, if you tell
Pi Camera to resize the capture to Inception V3's required 299x299, it actually
rounds up to 320x304. The captured image has to be cropped before passed to the
Edge TPU.

Here, I have filled those gaps by giving a few additional demos:

1. `classify_capture.py`: This is the original official demo, kept for
reference. It does not check the model input size against Pi Camera's capable
frame size. Models of input 224x224 work fine (because 224 is a multiple of 32
and 16), but other input sizes may not.

2. `classify_capture_crop.py`: This preemptively checks the model input size
against Pi Camera's capable frame size. Frame size is scaled up if they don't
match. The captured image is cropped, if necessary, before passed to the Edge
TPU. It should work for all model input sizes.

3. `detect_capture_crop.py`: This does *object detection*. It uses *overlay* to
put rectangles on the preview. It also scales up frame size and crops captures
if necessary. There is one nagging thing, though: every time the overlay is
refreshed, Pi Camera spills out this error:

    ```
    picamera.exc.PiCameraMMALError: no buffers available: Resource temporarily unavailable; try again later
    ```

    [I could not](https://github.com/waveform80/picamera/issues/448)
    [find](https://github.com/waveform80/picamera/issues/320)
    [any solution](https://github.com/waveform80/picamera/issues/393) ...

    ... until I came across [AIY Vision Kit's source code](https://github.com/google/aiyprojects-raspbian),
    a part of which I have incorporated into the following.

4. `detect_capture_crop_with_annotator.py`: It does object detection as above,
   but the overlay logic is separated into an `Annotator` class, which makes the
   code a little cleaner. The `Annotator` class is taken from the AIY Vision Kit
   project. It patches the picamera package to stop it from spilling out errors
   mentioned above. Remember to set `PYTHONPATH` to use the `Annotator` class.
