from matplotlib import pyplot as plt
import time
import os
from ..DebugSwitcher import is_plot

PLOT_ROW = 20
PLOT_COL = 2
plot_index = 0

plt.figure(figsize=(20, 200))


def next_idx():
    global plot_index
    plot_index += 1
    return plot_index


def reset():
    global plot_index
    plot_index = 0
    plt.close()


def next_idx():
    global plot_index
    plot_index += 1
    return plot_index


def reset():
    global plot_index
    plot_index = 0


def id(index):
    return PLOT_ROW * 100 + PLOT_COL * 10 + index


def subImage(src, index=0, figsize=None, plot_row=None, plot_col=None, title=None, cmap=None):
    if not is_plot:
        return -1
    global PLOT_ROW, PLOT_COL
    if index == 0:
        raise Exception("Index should be specified")
    if plot_row is not None:
        PLOT_ROW = plot_row
    if plot_col is not None:
        PLOT_COL = plot_col
    if figsize is not None:
        plt.figure(figsize=figsize)
    if index > PLOT_ROW * PLOT_COL:
        print("Index over plot size range,resize plot size.\n")
        PLOT_ROW += 1
    if src is None:
        raise Exception("Image is None.Plot error.")
    # print(id(index))
    # if index > 9:
    #    plot.figure(figsize=(20, 80))
    #    index %= 9
    # plot.subplot(id(index))
    plt.subplot(PLOT_ROW, PLOT_COL, index)
    if cmap is not None:
        plt.imshow(src, cmap=cmap)
    else:
        plt.imshow(src)
    plt.title(title)


def plot(src, index, title):
    global PLOT_ROW, PLOT_COL
    plt.subplot(PLOT_ROW, PLOT_COL, index)
    plt.plot(src)
    plt.title(title)


def save():
    str_time = time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time()))
    is_exists = os.path.exists('./output')
    # 判断结果
    if not is_exists:
        os.makedirs('./output')
    plt.savefig('./output/' + str_time + '.png')
    reset()


def show(save=False):
    if not is_plot:
        return -1
    if save:
        str_time = time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time()))
        plt.savefig('./output/' + str_time + '.png')
    reset()
    plt.show()
    plt.close()