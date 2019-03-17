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

Here, I have filled those gaps by giving two additional demos:

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

    [I cannot find](https://github.com/waveform80/picamera/issues/448)
    [any solution](https://github.com/waveform80/picamera/issues/320)
    [after investigation.](https://github.com/waveform80/picamera/issues/393)
    It seems to be an unfixed bug in the picamera package.
    Fortunately, it does not seem to have any ill effects otherwise.
    Object detection works fine, the program keeps on running. I will leave it
    at that.
