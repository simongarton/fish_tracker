from random import randint
import numpy as np
import imutils
import cv2 as cv
import sys
import shutil
import os

MAX_DELTA = 75  # how far a fish might move before I decide it's too fast for a fish
MAX_AGE = 50  # how much I can build a fishes age / confidence up before it peaks

SCAN_COLOR = (255, 255, 255)
SCAN_LINE_WIDTH = 1

CONTOUR_COLOR = (0, 255, 0)

FISH_COLOR = (255, 255, 255)
FISH_LINE_WIDTH = 2

TRAIL_COLOR = (50, 255, 0)
TRAIL_LINE_WIDTH = 2
TRAIL_POINT_SIZE = 2
MAX_TRAIL_POINTS = 400
TRAIL_WIDTH = 2
SMOOTH_TRAIL_WIDTH = 3

DRAW_BOX = False  # draw bounding box of fish
DRAW_TRAIL_POINTS = False  # draw trail as points
DRAW_TRAIL = False  # draw trail as line - ends up quite jagged
DRAW_SMOOTH_TRAIL = True  # draw trail as smoothed (average of last 10) points
DRAW_ID = True  # draw the fish ID
DRAW_STATS = True  # draw the stats panel bottom left

SHOW_VISION = False  # show a separate window for the vision analysis
SHOW_DELTA = False  # show the image deltas
SHOW_THRESH = False  # show the dilated threshold analysis
show_panels = True  # show the nice sliding panels. toggle with 't'

width = 0
height = 0
fish_count = 0
image_count = 0  # index screenshots
slide = 0  # slide index for the sliding panels

font = cv.FONT_HERSHEY_SIMPLEX


def create_fish(x, y, w, h):
    return {
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'age': 0,
        'points': []
    }


def draw_possible_fish(gray, possible_fish):
    # this might be a fish, I will confirm later
    x = possible_fish['x']
    y = possible_fish['y']
    w = possible_fish['w']
    h = possible_fish['h']
    cv.rectangle(gray, (x, y), (x + w, y + h), SCAN_COLOR, SCAN_LINE_WIDTH)


def close_enough(fish, possible_fish):
    # is this possible fish close enough to this real one to match ? pythagoras
    delta_x = (fish['x'] - possible_fish['x']) ** 2
    delta_y = (fish['y'] - possible_fish['y']) ** 2
    dist = (delta_x + delta_y) ** 0.5
    return dist < MAX_DELTA


def average_point(fish, range):
    # average out the last n points. used to show the id
    subset = fish['points'][-range:] if len(
        fish['points']) > range else fish['points']
    n = len(subset)
    x = y = 0
    for point in subset:
        x = x + point[0]
        y = y + point[1]
    return (int(x * 1.0 / n), int(y * 1.0 / n))


def smooth(points):
    # smooth out the trail points
    if len(points) < 10:
        return points
    total_x = 0
    total_y = 0
    for index in range(0, 10):
        total_x = total_x + points[index][0]
        total_y = total_y + points[index][1]
    smoothed = []
    for index in range(10, len(points)):
        total_x = total_x + points[index][0] - points[index-10][0]
        total_y = total_y + points[index][1] - points[index-10][1]
        smoothed.append((int(total_x * 1.0/10), int(total_y * 1.0/10)))
    return smoothed


def draw_fish(frame, fish):
    # this is not working well, it's picking up only tiny changes when the fish is moving slowly ...
    if DRAW_BOX:
        cv.rectangle(
            frame, (fish['x'], fish['y']), (fish['x'] + fish['w'], fish['y'] + fish['h']), FISH_COLOR, FISH_LINE_WIDTH)
    if DRAW_TRAIL_POINTS:
        for point in fish['points']:
            cv.circle(frame, point, 2, (0, 255, 0), -1)
    if DRAW_TRAIL:
        points = np.array(fish['points'])
        points = points.reshape((-1, 1, 2))
        cv.polylines(frame, [points], False, (0, 255, 0), TRAIL_WIDTH)
    if DRAW_SMOOTH_TRAIL:
        points = np.array(smooth(fish['points']))
        points = points.reshape((-1, 1, 2))
        cv.polylines(frame, [points], False, fish['color'], SMOOTH_TRAIL_WIDTH)


def draw_fish_id(frame, fish):
    global font
    if DRAW_ID and len(fish['points']) > 0:
        point = average_point(fish, 10)
        displaced_point = (int(point[0] + fish['w']/2),
                           int(point[1] + fish['h']/1))
        cv.putText(frame, str(fish['id']), displaced_point, font,
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
            fish['age'] = fish['age'] + \
                5 if fish['age'] < MAX_AGE else fish['age']
            return
    possible_fish['age'] = 5
    possible_fish['id'] = fish_count
    possible_fish['color'] = (
        randint(50,  255), randint(50,  255), randint(50,  255))
    fish_count = fish_count + 1

    fishes.append(possible_fish)


def setup_video(args):

    global width
    global height

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


def find_contours(frame, firstFrame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_smooth = cv.GaussianBlur(gray, (21, 21), 0)
    if len(firstFrame) == 0:
        firstFrame = gray_smooth

    frameDelta = cv.absdiff(firstFrame, gray_smooth)

    thresh = cv.threshold(frameDelta, 20, 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, None, iterations=10)  # this is hight

    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts, gray, gray_smooth, frameDelta, thresh


def manage_fishes(cnts, fishes):
    # We loop over all the bbs of the contours that I can find.
    # For each one, I check to see if it matches - with some approximation - a fish
    # If so, I add ten to the fish's value
    # If not, I create a new fish with value 5

    for c in cnts:
        if cv.contourArea(c) < 20:  # conf["min_area"]:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        if w < 20:
            w = 20
        if h < 10:
            h = 10
        possible_fish = create_fish(x, y, w, h)
        match_possible_fish_to_fishes(fishes, possible_fish)


def update_fishes(fishes, frame):
    active = 0
    for fish in fishes:
        fish['age'] = fish['age'] - 1
        if fish['age'] < 0:
            # fishes.remove(fish)
            continue
        active = active + 1

    for fish in fishes:
        if fish['age'] > 0:
            draw_fish(frame, fish)
            draw_fish_id(frame, fish)
    return active


def draw_stats(frame, fishes, active):
    global width
    global height
    global fish_count
    global font

    cv.rectangle(frame, (width - 250, height - 100),
                 (width, height), (0, 0, 0), -1)
    cv.putText(frame, 'active {}'.format(active), (width - 220, height - 60), font,
               1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, 'total   {}'.format(fish_count), (width - 220, height - 20), font,
               1, (255, 255, 255), 2, cv.LINE_AA)


def show_sliding_panels(frame, backtorgb, thresh, frameDelta):
    global slide

    quarter = int(width / 4)
    extra_width = width + quarter
    x = ((0 + slide) % extra_width) - quarter
    a = x if x >= 0 else 0
    w = quarter if x >= 0 else quarter + x
    working = backtorgb
    frame[0:height, a:a + w] = working[0:height, a:a + w]
    cv.putText(frame, 'contours and bb', (x, 30), font,
               1, (255, 255, 255), 2, cv.LINE_AA)
    x = ((quarter + slide) % extra_width) - quarter
    a = x if x >= 0 else 0
    w = quarter if x >= 0 else quarter + x
    working = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)
    frame[0:height, a:a + w] = working[0:height, a:a + w]
    cv.putText(frame, 'dilated threshold', (x, 30), font,
               1, (255, 255, 255), 2, cv.LINE_AA)
    x = ((2 * quarter + slide) % extra_width) - quarter
    a = x if x >= 0 else 0
    w = quarter if x >= 0 else quarter + x
    working = cv.cvtColor(frameDelta, cv.COLOR_GRAY2RGB)
    frame[0:height, a:a + w] = working[0:height, a:a + w]
    cv.putText(frame, 'image deltas', (x, 30), font,
               1, (255, 255, 255), 2, cv.LINE_AA)
    x = ((-quarter + slide) % extra_width) - quarter
    cv.putText(frame, 'result', (x, 30), font,
               1, (255, 255, 255), 2, cv.LINE_AA)
    slide = slide + 10


def run(args):

    global image_count
    global width
    global height
    global font
    global show_panels

    bad_frames = 0

    cap, out = setup_video(args)

    fishes = []

    firstFrame = []
    while True:
        ret, frame = cap.read()
        if not ret:
            bad_frames = bad_frames + 1
            if bad_frames > 100:
                print("Can't receive frame (stream end?). Exiting after 100 failures.")
                break
            continue

        cnts, gray, gray_smooth, frameDelta, thresh = find_contours(
            frame, firstFrame)

        manage_fishes(cnts, fishes)

        active = update_fishes(fishes, frame)

        # this is nice to see the actual changes
        if SHOW_DELTA:
            cv.imshow('delta', frameDelta)

        # threshold and dilate the changes
        if SHOW_THRESH:
            cv.imshow('thresh', thresh)

        if SHOW_VISION or show_panels:
            backtorgb = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
            cv.drawContours(backtorgb, cnts, -1,
                            CONTOUR_COLOR, SMOOTH_TRAIL_WIDTH)
            for c in cnts:
                (x, y, w, h) = cv.boundingRect(c)
                possible_fish = create_fish(x, y, w, h)
                draw_possible_fish(backtorgb, possible_fish)

            if SHOW_VISION:
                cv.imshow('vision-view', backtorgb)

            if show_panels:
                show_sliding_panels(frame, backtorgb, thresh, frameDelta)

        if DRAW_STATS:
            draw_stats(frame, fishes, active)

        cv.imshow('tank-view', frame)

        out.write(frame)

        key = cv.waitKey(1)
        if key == ord('t'):
            show_panels = not show_panels
        if key == ord('p'):
            print('saving images/{}.png'.format(image_count))
            cv.imwrite('images/{}.png'.format(image_count), frame)
            image_count = image_count + 1
        if key == ord('q'):
            break

        firstFrame = gray_smooth

    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    try:
        shutil.rmtree('images')
    except FileNotFoundError:
        pass
    os.makedirs('images')
    run(sys.argv)
