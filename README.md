# fish_tracker

OpenCV example to track fishes in tank.

Simon Garton  
simon.garton@gmail.com  
Easter weekend, 2022

![alt text](screenshots/overall.png 'Overall image showing fish and trails')

[Open CV](https://opencv.org/) is an impressively functional library for computer vision. I'd tinkered with it briefly a few years ago, but with a rainy Easter weekend I thought I'd have another look; and I was impressed.

I worked my way through the tutorials - including writing a little app to replace my eyes with the Eye of Sauron, which was briefly amusing. But I was intrigued by the motion detection stuff and thought it might be fun to write an app to track the fishes in my tank, showing how different species swim at different depths, and have different activity levels.

## Part 1 : motion tracking

Take this scene ... two yellow guppies diving down, 3 tetras hanging out.

![alt text](screenshots/original.png 'Original')

First of all, we need to work out what's changed. I gray-scale the original image, blur it a little to avoid noise, and then subtract it from the previous one.

![alt text](screenshots/deltas.png 'Image deltas')

You can just see the guppy outlines as they are moving actively; the tetras not so much. To emphasize, we threshold - make points black or white depending on if their brightness is over a threshold - and then dilate, expanding any colored areas so they are easier to spot.

![alt text](screenshots/threshold.png 'Threshold and dilate')

Now we generate contours around those areas, and build up a bounding box for the fish. I'm now superimposing these on a grey scale of the original image.

![alt text](screenshots/contours.png 'Contours and bounding box')

Finally I track the fish - setting up an array of actual fish, and for each new bounding box, try and work out if it's the same as a known fish or a new one. For each actual fish, record the last 100 positions and smooth it out as a trail, plus show the fish's id, all on the original stream.

![alt text](screenshots/plotted.png 'Plotted')

### Problems

This is my biggest problem. I have one fish on screen, but the head and tail are moving more than the middle, so I end up with two contacts, and not a single fish.

![alt text](screenshots/two-contacts.png 'Two contacts')

Reducing these to one fish - and not confusing two fish swimming close to each other - remains unsolved.

As a similar problem, when fish swim fast, they can move more than a fish length in a frame, and this means I lose track of the fish and create a second one.

## Part 2 : Discrimination

Now I need to figure out which species a fish is, so I can track them differently. I have red/blue tetras, yellow guppies, and black/brown tetras, all similar sizes.

I fairly quickly was able to snapshot 100x100 pixels around each fish ...

![alt text](screenshots/classify.png 'Classify as tetra')

... but the next bit is hard. I want some way of getting the most dominant color, or counting how many pixels of each color there are ... but I need to generate contours and apply a mask first. With that done, I ran a rough count of how many red, blue and green pixels there were - splitting into each channel and assuming a limit - and came up with a very approximate classification. And it sort of works ...

![alt text](screenshots/classified.png 'Tetras, plants and guppies')
