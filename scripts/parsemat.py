import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication, QMenu, QMenuBar, QAction, QFileDialog
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QBrush, QFont
import scipy.interpolate as interpolate
from PyQt5.QtCore import Qt, QPoint


fname = os.getcwd() + "/maps/track2.mat"
print(fname)
mat = sio.loadmat(fname)
centerline = mat['track2'][0,0][2]
waypoints = centerline.T
px_p_meter = 100

fig, ax1= plt.subplots(nrows = 1, ncols = 1, figsize=(10,10))
points = 3.5*np.array(waypoints)
print(f"wpts {waypoints}")
print(f"scaled:{points}")
points = points - np.mean(points, axis = 0).reshape(-1,2)
print(points)
points = np.concatenate((points, points[0:1,:]), axis=0)
print(points)
ax1.scatter(points[:,0], points[:,1])

pointvals = np.arange(len(points))
#points = np.vstack((points, points[0,:].reshape(-1,1).T))
x_int = interpolate.CubicSpline(pointvals, points[:,0], bc_type = 'periodic')
y_int = interpolate.CubicSpline(pointvals, points[:,1], bc_type = 'periodic')
x_i = x_int(np.linspace(0, len(points)-1, 2000))
y_i = y_int(np.linspace(0, len(points)-1, 2000))
#coords = np.array([x_vals,y_vals])
# tck, u = interpolate.splprep([points[:,0], points[:,1]], s=0.003)
# x_i, y_i = interpolate.splev(np.linspace(0, 1, 1000), tck)
points = np.vstack([x_i, y_i]).T
plt.plot(x_i,y_i)
plt.show()
print("Done showing")
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
win = Window()
name = QFileDialog.getSaveFileName(win, "Save Waypoints", "../dmaracing/maps", "CSV(*.csv)")
np.savetxt(name[0], points, delimiter = ', ')
print('done')