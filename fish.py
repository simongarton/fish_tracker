from doctest import FAIL_FAST
from turtle import pos
import numpy as np
import imutils
import cv2 as cv
import sys

MAX_DELTA = 75  # how far a fish might move before I decide it's too fast for a fish

SCAN_COLOR = (255, 255, 255)
SCAN_LINE_WIDTH = 1

FISH_COLOR = (255, 255, 255)
FISH_LINE_WIDTH = 2

TRAIL_COLOR = (50, 255, 0)
TRAIL_LINE_WIDTH = 2
TRAIL_POINT_SIZE = 2
MAX_TRAIL_POINTS = 100
DRAW_TRAIL_POINTS = False
DRAW_TRAIL = True
DRAW_ID = True

SHOW_VISION = False


bad_frames = 0
fish_count = 0


def make_fish(x, y, w, h):
    return {
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'age': 0,
        'points': []
    }


def draw_possible_fish(gray, possible_fish):
    x = possible_fish['x']
    y = possible_fish['y']
    w = possible_fish['w']
    h = possible_fish['h']
    cv.rectangle(gray, (x, y), (x + w, y + h), SCAN_COLOR, SCAN_LINE_WIDTH)


def close_enough(fish, possible_fish):
    delta_x = (fish['x'] - possible_fish['x']) ** 2
    delta_y = (fish['y'] - possible_fish['y']) ** 2
    dist = (delta_x + delta_y) ** 0.5
    # print('{}.{} {}.{} = {}'.format(
    #     fish['x'], fish['y'], possible_fish['x'], possible_fish['y'], dist))
    return dist < MAX_DELTA


def draw_fish(frame, fish):
    # draw the bounding box of the fish. this is not working well, it's picking up only tiny changes
    # when the fish is moving slowly ...
    cv.rectangle(
        frame, (fish['x'], fish['y']), (fish['x'] + fish['w'], fish['y'] + fish['h']), FISH_COLOR, FISH_LINE_WIDTH)
    # draw the last 100 trail points
    if DRAW_TRAIL_POINTS:
        for point in fish['points']:
            cv.circle(frame, point, 2, (0, 255, 0), -1)
    # draw the last 100 trail points as a line
    if DRAW_TRAIL:
        points = np.array(fish['points'])
        points = points.reshape((-1, 1, 2))
        cv.polylines(frame, [points], False, (0, 255, 0), 1)
    # draw name
    if DRAW_ID:
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, str(fish['id']), (fish['x'], fish['y']), font,
                   1, (255, 255, 255), 2, cv.LINE_AA)


def match_possible_fish_to_fishes(fishes, possible_fish):
    global fish_count
    for fish in fishes:
        if close_enough(fish, possible_fish):
            point = (fish['x'], fish['y'])
            fish['points'].append(point)
            if len(fish['points']) > MAX_TRAIL_POINTS:
                fish['points'].remove(fish['points'][0])
            fish['x'] = possible_fish['x']
            fish['y'] = possible_fish['y']
            fish['age'] = fish['age'] + 5 if fish['age'] < 100 else fish['age']
            return
    possible_fish['age'] = 5
    possible_fish['id'] = fish_count
    fish_count = fish_count + 1

    fishes.append(possible_fish)


def setup_video(args):

    if len(args) == 1:
        # WebCam : 0 is built in, 1 is USB
        cap = cv.VideoCapture(1)
    else:
        # Stream from MP4
        cap = cv.VideoCapture(args[1])

    if not cap.isOpened():
        print("Cannot open video stream")
        exit()

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)

    out = cv.VideoWriter(
        'out/out.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

    return cap, out


def find_contours(frame, firstFrame, avg):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_smooth = cv.GaussianBlur(gray, (21, 21), 0)
    if len(firstFrame) == 0:
        firstFrame = gray_smooth

    if avg is None:
        avg = gray.copy().astype("float")
        return [], gray, gray_smooth, avg

    frameDelta = cv.absdiff(firstFrame, gray_smooth)

    cv.accumulateWeighted(gray, avg, 0.5)

    thresh = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, None, iterations=1)

    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts, gray, gray_smooth, avg


def manage_fishes(cnts, fishes, gray):
    # We loop over all the bbs of the contours that I can find.
    # For each one, I check to see if it matches - with some approximation - a fish
    # If so, I add ten to the fish's value
    # If not, I create a new fish with value 5

    for c in cnts:
        if cv.contourArea(c) < 20:  # conf["min_area"]:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        possible_fish = make_fish(x, y, w, h)
        draw_possible_fish(gray, possible_fish)
        match_possible_fish_to_fishes(fishes, possible_fish)


def update_fishes(fishes, frame):
    for fish in fishes:
        fish['age'] = fish['age'] - 1
        if fish['age'] < 0:
            fishes.remove(fish)
            continue

        draw_fish(frame, fish)


def run(args):

    cap, out = setup_video(args)

    fishes = []

    firstFrame = []
    avg = None
    while True:
        ret, frame = cap.read()
        if not ret:
            bad_frames = bad_frames + 1
            if bad_frames > 100:
                print("Can't receive frame (stream end?). Exiting after 100 failures.")
                break
            continue

        cnts, gray, gray_smooth, avg = find_contours(frame, firstFrame, avg)

        manage_fishes(cnts, fishes, gray)

        update_fishes(fishes, frame)

        cv.imshow('tank-view', frame)
        if SHOW_VISION:
            cv.imshow('vision-view', gray)
        out.write(frame)

        if cv.waitKey(1) == ord('q'):
            break

        firstFrame = gray_smooth

    # When everything done, release the capture
    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    run(sys.argv)
