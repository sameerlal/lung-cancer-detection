"""
Module for plotting 3D images and other useful plotting-related functions.

ECE 4250 Final Project.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
April 21th, 2018.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_slices(img_arr, show=True):
    """
    Plot each slice of a 3D image, showing one slice at a time (use a and d keys to switch between slices).

    :param img_arr: 3D image array to plot.
    :param show: Whether or not to show the plot in addition to plotting the image.
    :return: None.
    """
    # Credit:
    # https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
    print('Getting subplots')
    fig, ax = plt.subplots()
    ax.img_arr = img_arr
    ax.index = 0
    print('showing image')
    vmin = np.min(img_arr)
    vmax = np.max(img_arr)
    ax.imshow(img_arr[ax.index], cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title('Current depth: {} (Press \'a\' or \'d\' to change)'.format(ax.index))
    print('calling mpl_connect')
    fig.canvas.mpl_connect('key_press_event', _process_key)
    if show:
        print('calling show')
        plt.show()
        print('finished show')


def _process_key(event):
    """Process key press event (used in plot_slices)."""
    if event.key == 'a':
        _change_slice(-1)
        plt.gcf().canvas.draw()
    elif event.key == 'd':
        _change_slice(1)
        plt.gcf().canvas.draw()


def _change_slice(delta):
    """Change the slice index by the given value (used in _process_key)."""
    ax = plt.gca()
    depth = ax.img_arr.shape[0]
    ax.index = (ax.index + delta) % depth  # Note: mod returns a positive number in Python.
    ax.images[0].set_array(ax.img_arr[ax.index])
    ax.set_title('Current depth: {}'.format(ax.index))
