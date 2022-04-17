import numpy as np
import imutils
import cv2 as cv
import json


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
    cv.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)


def close_enough(fish, possible_fish):
    delta_x = (fish['x'] - possible_fish['x']) ** 2
    delta_y = (fish['y'] - possible_fish['y']) ** 2
    dist = (delta_x + delta_y) ** 0.5
    print('{}.{} {}.{} = {}'.format(
        fish['x'], fish['y'], possible_fish['x'], possible_fish['y'], dist))
    return dist < 100


def draw_fish(frame, fish):
    col_index = fish['age'] * 10
    if col_index > 255:
        col_index = 255
    color = (col_index, 0, 0) if fish['age'] > 0 else (50, 50, 50)
    # cv.rectangle(
    #     frame, (fish['x'], fish['y']), (fish['x'] + fish['w'], fish['y'] + fish['h']), color, 5)
    for point in fish['points']:
        cv.circle(frame, point, 2, (0, 255, 0), -1)


def match_possible_fish_to_fishes(fishes, possible_fish):
    for fish in fishes:
        if close_enough(fish, possible_fish):
            point = (fish['x'], fish['y'])
            fish['points'].append(point)
            if len(fish['points']) > 100:
                fish['points'].remove(fish['points'][0])
            fish['x'] = possible_fish['x']
            fish['y'] = possible_fish['y']
            fish['age'] = fish['age'] + 5 if fish['age'] < 100 else fish['age']
            return
    possible_fish['age'] = 5
    fishes.append(possible_fish)


def run():

    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    conf = json.load(open('conf.json'))

    fishes = []

    firstFrame = []
    avg = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_smooth = cv.GaussianBlur(gray, (21, 21), 0)
        if len(firstFrame) == 0:
            firstFrame = gray_smooth

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        frameDelta = cv.absdiff(firstFrame, gray_smooth)

        cv.accumulateWeighted(gray, avg, 0.5)

        thresh = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]
        thresh = cv.dilate(thresh, None, iterations=1)

        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

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

        for fish in fishes:
            fish['age'] = fish['age'] - 1
            if fish['age'] < 0:
                fishes.remove(fish)
                continue

            draw_fish(frame, fish)

        cv.imshow('actual', frame)
        # cv.imshow('possible', gray)
        if cv.waitKey(1) == ord('q'):
            break

        firstFrame = gray_smooth

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    run()
