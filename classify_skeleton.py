import time
from enum import Enum, auto
from multiprocessing import Pipe, Process
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera import PiCamera
from picamera.array import PiRGBArray

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


class Instruction(Enum):
    LEFT = auto()
    RIGHT = auto()
    STRAIGHT = auto()
    STOP = auto()
    IDLE = auto()


def make_threshold_img(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    THRESHOLD = 100
    MAX_VAL = 255
    _, threshold_img = cv2.threshold(
        gray_img, THRESHOLD, MAX_VAL, cv2.THRESH_BINARY
    )

    # Flood fill with seeds located on a rectangle a little smaller than the window
    SEED_STEP = 10
    DISTANCE_FROM_EDGE = 20
    height, width = threshold_img.shape[:2]
    top_seeds = [(x, DISTANCE_FROM_EDGE) for x in range(0, width, SEED_STEP)]
    left_seeds = [(DISTANCE_FROM_EDGE, y) for y in range(0, height, SEED_STEP)]
    right_seeds = [
        (x, height - DISTANCE_FROM_EDGE) for x in range(0, width, SEED_STEP)
    ]
    bottom_seeds = [
        (width - DISTANCE_FROM_EDGE, y) for y in range(0, height, SEED_STEP)
    ]

    for seed in top_seeds + left_seeds + right_seeds + bottom_seeds:
        WHITE = 255
        mask = np.zeros((height + 2, width + 2), np.uint8)
        cv2.floodFill(threshold_img, mask, seed, WHITE)

    return threshold_img


def detect_shape(color_img):
    """
    PART 1
    Isolate (but do not detect) the arrow/stop sign using image filtering techniques.
    Return a mask that isolates a black shape on white paper

    Checkoffs: None for this part!
    """

    threshold_img = make_threshold_img(color_img)

    """
    END OF PART 1
    """

    # Find contours in the filtered image.
    contours, _ = cv2.findContours(
        threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

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

    # Small contours are likely noise, large contour is likely the whole window
    MIN_AREA = 1000
    MAX_AREA = 100_000

    traffic_sign = None
    sign_idx = -1

    for idx, contour in enumerate(contours):
        if MIN_AREA <= cv2.contourArea(contour) <= MAX_AREA:
            traffic_sign = contour
            sign_idx = idx
            break

    if traffic_sign is None:
        return (sign_idx, Instruction.IDLE)

    # Convexity
    area = cv2.contourArea(traffic_sign)
    hull = cv2.contourArea(cv2.convexHull(traffic_sign))

    try:
        convexity = area / hull
    except ZeroDivisionError:
        convexity = 0

    CUTOFF = 0.9

    if convexity > CUTOFF:
        # Stop sign detected
        return (sign_idx, Instruction.STOP)

    moments = cv2.moments(traffic_sign)

    try:
        mx = int(moments["m10"] / moments["m00"])
        my = int(moments["m01"] / moments["m00"])
    except ZeroDivisionError:
        (mx, my) = (0, 0)

    # Bounding box
    x, y, w, h = cv2.boundingRect(traffic_sign)

    tilt_ratio = abs(x - mx) / w

    MIDDLE_TILT = 0.5
    TOLERANCE = 0.05

    if tilt_ratio > MIDDLE_TILT + TOLERANCE:
        # Arrow is pointing right
        return (sign_idx, Instruction.RIGHT)
    elif tilt_ratio < MIDDLE_TILT - TOLERANCE:
        # Arrow is pointing left
        instructions = "left"
        return (sign_idx, Instruction.LEFT)

    return (sign_idx, Instruction.STRAIGHT)
    """
    END OF PART 2
    """


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

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Motor constants
ESC_CONTROLLING_PIN = 7
MOTOR_FREQUENCY = 60
NEUTRAL_MOTOR_SIGNAL = 8.0
FULL_FORWARD_SIGNAL = 10.3
FULL_BACK_SIGNAL = 5.7


class Motor:
    PIN = 37
    FREQ = 60
    BACK = 10.3
    FORWARD = 5.7
    NEUTRAL = 8


class Servo:
    PIN = 8
    FREQ = 60
    STRAIGHT = 7.6
    LEFT = 8
    RIGHT = 5


class Car:
    TURN_TIME = 2

    def __init__(self):
        GPIO.setup(Motor.PIN, GPIO.OUT)
        GPIO.setup(Servo.PIN, GPIO.OUT)

        self.motor = GPIO.PWM(Motor.PIN, Motor.FREQ)
        self.servo = GPIO.PWM(Servo.PIN, Servo.FREQ)

        self.motor.start(Motor.NEUTRAL)
        self.servo.start(Servo.STRAIGHT)

        self.straight()

    def close(self):
        self.motor.stop()
        self.servo.stop()
        GPIO.cleanup()

    def __del__(self):
        self.close()

    def arm_motor(self):
        self.motor.ChangeDutyCycle(Motor.NEUTRAL)
        input("Enter anything to continue: ")

    def stop(self):
        self.motor.ChangeDutyCycle(Motor.NEUTRAL)
        self.servo.ChangeDutyCycle(Servo.STRAIGHT)

    def forward(self):
        self.motor.ChangeDutyCycle(Motor.FORWARD)
        self.servo.ChangeDutyCycle(
            Servo.STRAIGHT
        )  # Motor somehow influences the servo

    def back(self):
        self.motor.ChangeDutyCycle(Motor.BACK)

    def turn(self, cycle_value):
        self.servo.ChangeDutyCycle(cycle_value)
        time.sleep(self.TURN_TIME)

    def straight(self):
        self.turn(Servo.STRAIGHT)

    def left(self):
        self.turn(Servo.LEFT)

    def right(self):
        self.turn(Servo.RIGHT)

    def calibrate_motor(self):
        try:
            while True:
                self.motor.ChangeDutyCycle(float(input("motor: ")))
        except KeyboardInterrupt:
            pass

    def calibrate_servo(self):
        try:
            while True:
                self.servo.ChangeDutyCycle(float(input("servo: ")))
        except KeyboardInterrupt:
            pass


car = Car()

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
            print(frame_count)
            if frame_count == 1:
                print(img.shape)

            checkoff_img = None

            contour_idx, instruction = detect_shape(img)

            threshold_img = make_threshold_img(img)
            contours, _ = cv2.findContours(
                threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            formatted_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)

            if contours and contour_idx != -1:
                moments = cv2.moments(contours[contour_idx])
                try:
                    moment = (
                        int(moments["m10"] / moments["m00"]),
                        int(moments["m01"] / moments["m00"]),
                    )
                except ZeroDivisionError:
                    moment = (0, 0)

                x, y, w, h = cv2.boundingRect(contours[contour_idx])
                midline = ((x + w // 2, y), (x + w // 2, y + h))

                checkoff_img = part2_checkoff(
                    formatted_img,
                    contours,
                    contour_idx,
                    moment,
                    midline,
                    f"{instruction}",
                )
            else:
                checkoff_img = cv2.putText(
                    formatted_img,
                    f"{instruction}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("RPi Camera Feed", checkoff_img)

            """
            PART 4
            1. Figure out the values of your motor and Servo PWMs for each instruction
               from `detect_shape()`.
            2. Assign those values as appropriate to the motor and Servo pins. Remember
               that an instruction of "idle" should leave the car's behavior UNCHANGED.

            Checkoffs: Show the leads your working car!
            """

            print(instruction)

            if instruction == Instruction.IDLE:
                pass
            elif instruction == Instruction.STRAIGHT:
                car.straight()
                car.forward()
            elif instruction == Instruction.LEFT:
                car.left()
                car.forward()
            elif instruction == Instruction.RIGHT:
                car.right()
                car.forward()
            elif instruction == Instruction.STOP:
                car.stop()

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
cv2.destroyAllWindows()
vs.stop()
