import glob
import cv2 as cv
from settings import *
from bbox import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd

objects = [0, 0, 0]  # cars, buses, trucks
ns = ps = 0  # negative and positive samples
squares = [dict()] * 3  # dicts with squares of objects
max_area = 0
for img_path in glob.glob(dataset_path + '\\*.jpg'):
    txt = open(img_path[:-4] + '.txt', 'r').readlines()
    if len(txt) == 0:
        ns += 1
        continue
    else:
        ps += 1
    for bbox in txt:
        bbox = read_from_YOLO(bbox)
        objects[bbox.get_label()] += 1

        cur_dict = squares[bbox.get_label()].copy()
        area = ((bbox.get_area() - 1) // 2000 + 1) * 2000
        if area > max_area:
            max_area = area
        if cur_dict.get(area) is None:
            cur_dict[area] = 1
        else:
            cur_dict[area] += 1
        squares[bbox.get_label()] = cur_dict


x_labels = []
y_labels = []
for n, css in enumerate(squares):
    x_labels.append(np.array(sorted(css)))
    counts = []
    for area in x_labels[n]:
        counts.append(squares[n][area])
    y_labels.append(counts)


# x_label = np.arange(0, max_area, 2000)
# y_label = np.zeros((x_label.size, 3), int)
# for count, area in enumerate(x_label):
#     line = []
#     for n in range(3):
#         if squares[n].get(area) is None:
#             line.append(0)
#         else:
#             line.append(squares[n][area])
#     y_label[count] = line


fig = plt.figure(figsize=(10, 8), constrained_layout=True)
fig.suptitle(name_for_stats, fontsize=30)
gs = fig.add_gridspec(4, 2)

ax1 = fig.add_subplot(gs[0])
ax1.pie([ns, ps], labels=["NS", "PS"], shadow=True, autopct='%1.1f%%')
ax1.set_title(f"{ns + ps} samples")

ax2 = fig.add_subplot(gs[0, 1])
ax2.pie(objects, labels=["Cars", "Buses", "Trucks"], shadow=True, autopct='%1.1f%%')
ax2.set_title(f"{sum(objects)} objects")

# ax3 = fig.add_subplot(gs[1, :])
#
# X_Y_Spline = make_interp_spline(x_label, y_label)
# X_ = np.linspace(x_label.min(), x_label.max(), 500)
# Y_ = X_Y_Spline(X_)
#
# ax3.plot(X_, Y_, lw=2, label=['Cars', 'Buses', 'Trucks'])
# ax3.legend(loc='upper right')
# ax3.set_ylabel('Count of objects')
# ax3.set_xlabel('Square in pixels')
# ax3.set_xlim(xmin=x_label[0], xmax=x_label[-1])
# ax3.set_ylim(ymin=0)
labels = ['Cars', 'Buses', 'Trucks']
for i in range(3):
    axis = fig.add_subplot(gs[i + 1, :])
    X_Y_Spline = make_interp_spline(x_labels[i], y_labels[i])
    X_ = np.linspace(x_labels[i].min(), x_labels[i].max(), 500)
    Y_ = X_Y_Spline(X_)
    axis.plot(X_, Y_, lw=2, label=labels[i])
    axis.legend(loc='upper right')
    axis.set_ylabel('Count of objects')
    axis.set_xlabel('Square in pixels')
    axis.set_xlim(xmin=x_labels[i][0], xmax=x_labels[i][-1])
    axis.set_ylim(ymin=0)

plt.show()
