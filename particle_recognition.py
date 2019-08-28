from video_processing import Video
import numpy as np
import math as m
from scipy.optimize import curve_fit
from data import data as dddd

import matplotlib.pyplot as plt

from lmfit import Model
import copy


def h(x): return 0.5 * (np.sign(x) + 1)


def step(x, a, b0, b1): return (b1 - b0) * (np.sign(x - a) + 1) + b0


def linear(x, a, b):
    return a * x + b


def chute(x, a0, a1, b0, b1):
    return b0 * h(a0 - x) + b1 * h(x - a1) + ((b1 - b0) / (a1 - a0) * (x - a0) + b0) * h(x - a0) * h(a1 - x)


def t2i(boo):
    if boo:
        return 1
    else:
        return 0


def find_step(data):
    return np.argmax([m.fabs(data[i] - data[i + 2]) for i in range(len(data) - 2)])


def is_np(data, mx=2e-03, show=False):
    xdata = np.arange(len(data))

    popt, pcov = curve_fit(step, xdata, data, p0=[find_step(data), 0, -5e-04], epsfcn=0.1)

    popt2, pcov2 = curve_fit(step, xdata, data, p0=[10, 0, -5e-04], epsfcn=0.1)
    print('guess: {}'.format(find_step(data)))

    squares = sum([(step(i, *popt) - data[i]) ** 2 for i in xdata])
    squares2 = sum([(step(i, *popt2) - data[i]) ** 2 for i in xdata])

    lpopt, lpcov = curve_fit(linear, xdata, data, p0=[1e-4, 0], epsfcn=0.1)
    lsquares = sum([(linear(i, *lpopt) - data[i]) ** 2 for i in xdata])

    if show:
        print('a, b: {}'.format(popt))
        #    print(pcov)
        print('delta: {}'.format(m.fabs(popt[2] - popt[1])))
        print('step: {}'.format(squares))
        print('step2: {}'.format(squares2))

        print('linear {}: '.format(lsquares))
        print(2 * squares < lsquares)
        print('variance: {}'.format(np.var(data)))

        fix, axes = plt.subplots()
        axes.plot(data, 'b-', label='data')
        axes.plot(xdata, step(xdata, *popt), 'r-')
        axes.plot(xdata, step(xdata, *popt2), 'k-')
        axes.plot(xdata, linear(xdata, *lpopt), 'g-')

    return (m.fabs(popt[2] - popt[1]) > 1e-04 and 2 * squares < lsquares) or (
                m.fabs(popt[2] - popt[1]) > 5e-04 and squares < lsquares) or (np.abs(data[-1]) > mx)


def is_np_new(inten, show=False):
    sigma = np.ones(len(inten))
    sigma[:3] = 10
    sigma[-3:] = 10

    x = np.arange(0, 1e-04, 1e-04 / len(inten))
    print(len(inten))
    print(len(x))
    model = Model(step)
    result = model.fit(inten, x=x, a=0.5e-04, b0=0, b1=-5e-04, weights=sigma)
    if show:
        fix, axes = plt.subplots()
        axes.plot(x, inten, 'b-', label='data')
        axes.plot(x, result.init_fit, 'k--')
        axes.plot(x, result.best_fit, 'r-')
        print(model.param_names)
        print(result.fit_report())


# def is_np(inten, treshold=3e-07, show=False):
#
#    xdata=np.arange(len(inten))
#    popt, pcov = curve_fit(chute, xdata, inten, p0=[5, 10, 0, -1e-04], epsfcn=0.1)
#    squares=sum([(chute(i, *popt)-inten[i])**2 for i in xdata])
#    if show:
#        print('a, b: {}'.format(popt))
#        #    print(pcov)
#        print('squares: {}'.format(squares))
#        print('variance: {}'.format(np.var(inten)))
#        fix, axes = plt.subplots()
#        axes.plot(inten,'b-', label='data')
#        axes.plot(xdata, chute(xdata, *popt), 'r-')  
#        
#    return m.fabs(popt[3]-popt[2])>3e-04

plt.close("all")
folder = 'C:/Users/jabuk/Documents/jaderka/ufe/data/'
main_folder = 'C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder = main_folder + '19_08_16_K4/'
# vyska, sirka, cas
file = 'norm_05_1'

video = Video(folder, file)
video.loadData()

# data=video._video[40:160, 1000:1200, :]
# del video
##plt.imshow(data[:,:,22]+mask*1e-02)
#
# mask=np.zeros(data[:,:,0].shape)
#
# for i in range(data.shape[0]):
#    print('done {}/{}'.format(i, data.shape[0]))
#    for j in range(data.shape[1]):
#        try:
#            mask[i,j]=t2i(is_np(data[i,j,:]))
#        except:
#            mask[i,j]=0
#            print('no fit')
#        
# print('done')
##
# plt.imshow(mask)

#
# fix, axes = plt.subplots()
# fixno, axesno = plt.subplots()
for i in range(len(dddd)):
    print(is_np(dddd[i], show=True))
    if i < 11:
        #        axes.plot(dddd[i])
        print(i)
    else:
        #        axesno.plot(dddd[i])
        print(i)

# print(is_np(data[0,0,:], show=True))
# print(is_np(data[68,135,:], show=True))
# print(is_np(data[50,135,:], show=True))
# plt.show()
