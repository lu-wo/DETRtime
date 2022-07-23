from itertools import count 
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.random import choice


def animate_prediction(y_pred, y_true, window_size, sample_len, fps=10, interval=10, dpi=200, fname='pred_animation.mp4'):
    """
    Plots the ground truth and prediction as mp4 video by sliding over the two prediction signals 
    y_pred: prediction vector (1d)
    y_true: ground truth vector (1d)
	window_size: size of sliding window displayed throughout the window (e.g. 100)
	sample_len: total length of the signal to displayed (e.g. 500)
	fps: frames per second 
	interval: time between frames 
	dpi: pixel density per inch 
	fname: path where to save the video (e.g. "./log/model01_pred01.mp4")
	-----------------------------------------------------------------------
	Expects 0 to be fixation, 1 to be saccade, 2 to be blink. 
    """
    l1 = y_pred
    l2 = y_true 
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    axes.set_ylabel("0: fix, 1: sac, 2: blk")
    axes.set_xlabel("time")
    # set limit for x and y axis
    axes.set_ylim(-0.5, 3)
    # style for plotting line
    plt.style.use("ggplot")
    # after every iteration
    x1, y1, y2 = [], [], []
    myvar = count(0, 1)
    DISPLAY = window_size
    axes.set_xlim(0, DISPLAY) 
    line1 = axes.plot([0], [0], color='red', label="prediction")
    line2 = axes.plot([0], [0], color="green", label='ground truth')  
    axes.legend()

    def animate(i):
        time = next(myvar)        
        x1.append(time)
        y1.append((l1[i]))
        y2.append((l2[i]))
        start = 0
        end = DISPLAY
        # move window
        if i > DISPLAY:
            start = i - DISPLAY
            end = i
            axes.set_xlim(start, end)
        line1 = axes.plot(x1[start: end], y1[start: end], color='red', label="prediction")
        line2 = axes.plot(x1[start: end], y2[start: end], color="green", label='ground truth')  
        
    # set ani variable to call the
    # function recursively
    anim = FuncAnimation(fig, animate, frames=sample_len-1, interval=interval)
    anim.save(fname, fps=fps, dpi=dpi)