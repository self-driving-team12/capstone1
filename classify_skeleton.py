import cv2
import numpy as np
import RPi.GPIO as GPIO
from threading import Thread
from multiprocessing import Process, Pipe
from queue import Queue
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
import copy

"""
DO NOT CHANGE THIS CLASS.
Parallelizes the image retrieval and processing across two cores on the Pi.
"""


class PiVideoStream:
    def __init__(self, resolution=(640, 480), framerate=32):
        self.process = None
        self.resolution = resolution
        self.framerate = framerate

    def start(self):
        pipe_in, self.pipe_out = Pipe()
        # start the thread to read frames from the video stream
        self.process = Process(
            target=self.update, args=(pipe_in,), daemon=True
        )
        self.process.start()
        return self

    def update(self, pipe_in):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = self.resolution
        self.camera.framerate = self.framerate
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False
        self.rawCapture = PiRGBArray(self.camera, size=self.resolution)
        self.stream = self.camera.capture_continuous(
            self.rawCapture, format="bgr", use_video_port=True
        )
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            pipe_in.send([self.frame])
            self.rawCapture.truncate(0)
            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                self.process.join()
                return

    def read(self):
        # return the frame most recently read
        if self.pipe_out.poll():
            return self.pipe_out.recv()[0]
        else:
            return None

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


print("[INFO] sampling MULTIPROCESSED frames from `picamera` module...")
vs = PiVideoStream(resolution=(640, 480)).start()
time.sleep(2.0)

"""
DO NOT CHANGE THIS FUNCTION.

Annotates your filtered image with the values you calculate.

PARAMETERS:
img -               Your filtered BINARY image, converted to BGR or
                    RGB form using cv2.cvtColor().

contours -          The list of all contours in the image.

contour_index -     The index of the specific contour to annotate.

moment -            The coordinates of the moment of inertia of
                    the contour at `contour_index`. Represented as an
                    iterable with 2 elements (x, y).

midline -           The starting and ending points of the line that
                    divides the contour's bounding box in half,
                    horizontally. Represented as an iterable with 2
                    tuples, ( (sx,sy) , (ex,ey) ), where `sx` and `sy`
                    represent the starting point and `ex` and `ey` the
                    ending point.

instruction -       A string chosen from "left", "right", "straight", "stop",
                    or "idle".
"""


def part2_checkoff(img, contours, contour_index, moment, midline, instruction):
    img = cv2.drawContours(img, contours, contour_index, (0, 0, 255), 3)
    img = cv2.circle(img, (moment[0], moment[1]), 3, (0, 255, 0), 3)

    img = cv2.line(img, midline[0], midline[1], (0, 0, 255), 3)

    img = cv2.putText(
        img,
        instruction,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return img


def detect_shape(color_img):
    """
    PART 1
    Isolate (but do not detect) the arrow/stop sign using image filtering techniques.
    Return a mask that isolates a black shape on white paper

    Checkoffs: None for this part!
    """

    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    THRESHOLD = 100
    MAX_VAL = 255
    _, threshold_img = cv2.threshold(
        gray_img, THRESHOLD, MAX_VAL, cv2.THRESH_BINARY
    )

    """
    END OF PART 1
    """

    # Create the color image for annotating.
    formatted_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)

    # Find contours in the filtered image.
    contours, hierarchy = cv2.findContours(
        threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return

    """
    PART 2
    1. Identify the contour with the largest area.
    2. Find the centroid of that contour.
    3. Determine whether the contour represents a stop sign or an arrow. If it's an
       arrow, determine which direction it's pointing.
    4. Set instruction to 'stop' 'idle' 'left' 'right' 'straight' depending on the output
    5. Use the part2_checkoff() helper function to produce the formatted image. See
       above for documentation on how to use it.

    Checkoffs: Send this formatted image to your leads in your team's Discord group chat.
    """

    # Only use large contours
    MIN_AREA = 5000
    large_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= MIN_AREA
    ]

    GREEN = (0, 255, 0)
    cv2.drawContours(formatted_img, large_contours, -1, GREEN, 3)

    key_points = []

    for contour in large_contours:
        area_and_hull = (
            cv2.contourArea(contour),
            cv2.contourArea(cv2.convexHull(contour)),
        )
        print(area_and_hull[0], end=", ")

        try:
            convexity = area_and_hull[0] / area_and_hull[1]
        except ZeroDivisionError:
            convexity = 0

        print(convexity, end=", ")

        moments = cv2.moments(contour)
        try:
            mx = int(moments["m10"] / moments["m00"])
            my = int(moments["m01"] / moments["m00"])
        except ZeroDivisionError:
            (mx, my) = (0, 0)
        print(f"({mx}, {my})", end=", ")
        key_points.append(cv2.KeyPoint(mx, my, 1))
    print()

    RED = (0, 0, 255)
    img_with_keys = cv2.drawKeypoints(
        formatted_img, key_points, 0, RED, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )
    cv2.imshow("Video", img_with_keys)

    # TODO: Use area to distinguish sign and paper background

    instruction = "idle"

    """
    END OF PART 2
    """
    return instruction


"""
PART 3
0. Before doing any of the following, arm your ESC by following the instructions in the
   spec. You only have to do this once. Than the range will be remembered by the ESC
1. Set up two GPIO pins of your choice, one for the ESC and one for the Servo.
   IMPORTANT: Make sure your chosen pins aren't reserved for something else! See pinout.xyz
   for more details.
2. Start each pin with its respective "neutral" pwm signal. This should be around 8% for both.
   The servo may be slightly off center. Fix this by readjusting the arm of the servo (unscrew it,
   set the servo to neutral, make the wheel point straight, then reattach the arm). The arm may still
   not be perfectly alighned so use the manual_pwm.py program to determine your Servo's best neutral
   position.
3. Start the motor at the full-forward position (duty cycle = 5.7).

NOTE: If you change the variable names pwm_m and pwm_s, you'll also need to update the
      cleanup code at the bottom of this skeleton.

Checkoffs: None for this part!
"""

pwm_m = None
pwm_s = None
print("started!")

"""
END OF PART 3
"""

"""
PART 4
1. 
"""

frame_count = 0
left_count = 0
right_count = 0
last_instruction = None

try:
    while True:
        if vs.pipe_out.poll():
            result = vs.read()
            img = cv2.rotate(result, cv2.ROTATE_180)

            frame_count += 1
            if frame_count == 1:
                print(img.shape)

            instruction = detect_shape(img)

            """
            PART 4
            1. Figure out the values of your motor and Servo PWMs for each instruction
               from `detect_shape()`.
            2. Assign those values as appropriate to the motor and Servo pins. Remember
               that an instruction of "idle" should leave the car's behavior UNCHANGED.

            Checkoffs: Show the leads your working car!
            """

            last_instruction = instruction

            """
            END OF PART 4
            """

            k = cv2.waitKey(3)
            if k == ord("q"):
                # If you press 'q' in the OpenCV window, the program will stop running.
                break
            elif k == ord("p"):
                # If you press 'p', the camera feed will be paused until you press
                # <Enter> in the terminal.
                input()
except KeyboardInterrupt:
    pass

# Clean-up: stop running the camera and close any OpenCV windows
pwm_m.stop()
pwm_s.stop()
GPIO.cleanup()
cv2.destroyAllWindows()
vs.stop()
