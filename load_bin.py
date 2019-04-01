# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:07:21 2019

@author: hlavacka
"""

import numpy as np
import matplotlib.pyplot as plt
import time

class VideoLoad(object):
    """Use to load and store video data.

    Object attributes:
    ------------------
        file_name
        video_stats - information about the loaded video data
        video - loaded video data

    Methods:
    --------
        loadData()
        loadBinVideoStats()
        loadBinVideo()
        videoCut() - cut image to a desired area
        frame(num=1) - show required frame
        animation(play_length=10, play_video_fps=24) - create a video animation
    """

    def __init__(self, file_name):
        """Holds video data with description.

        Creates an object to hold name of the source file with video data,
        information on how to analyse the video, information about the video
        stored in the file and the video data.

        Attributes:
        -----------
            file_name - name of the file with the video
            video_stats - information about the loaded video
            video - loaded video data

        Input parameters:
        -----------------
            file_name - name of the file with the video

        Return:
        -------
            object
        """
        self.file_name = file_name
        self.video_stats = None
        self.video = None
        self.view = None

    def __str__(self):
        """Return information string about the loaded video."""
        if self.video_stats is None:
            return 'No loaded video.'
        title = 'Video data:\n'
        fps = '    fps    = ' + str(self.video_stats[0]) + '\n'
        frame = '    frame  = (' + str(self.video_stats[1][0]) + ',' \
                                 + str(self.video_stats[1][1]) + ') px\n'
        length = '    length = {:0.2f} min\n'.format(self.video_stats[1][2]
                                                     /(self.video_stats[0]*60))
        return '\n' + title + fps + frame + length

    def __iter__(self):
        self.n = -1
        self.MAX = self.video.shape[2]-1
        return self

    def __next__(self):
        if self.n < self.MAX:
            self.n += 1
            return self.video[:, :, self.n]
        else:
            raise StopIteration

    def loadData(self):
        """Combine loading video and video information."""
        self.video_stats = self.loadBinVideoStats()
        self.video = self.loadBinVideo()

    def loadBinVideoStats(self):
        """Load video information.

        Loads information about the video to the object attribute.

        Return:
        -------
            integer - video_fps
            tuple - video_width, video_hight, video_length
        """
        suffix = '.tsv'
        with open(self.file_name+suffix, mode='r') as fid:
            file_content = fid.readlines()
        stats = file_content[1].split()
        video_length = len(file_content) - 1
        video_width = int(stats[1])
        video_hight = int(stats[2])
        video_fps = float(stats[4])*int(stats[5])
        self.view = [0, 0, video_width, video_hight]
        return [video_fps, [video_width, video_hight, video_length]]

    def loadBinVideo(self):
        """Load video in binary.

        Loads the binary video and saves it in a numpy array. The binary video
        is saved as doubles. Information about the size of frames and order of
        frames is in suplementary file with same name and ".tsv" suffix.
        The array consists of images ordered one after another.

        Return:
        -------
            numpy tensor of 3rd order
        """
        code_format = np.float64  # type double
        suffix = '.bin'
        with open(self.file_name+suffix, mode='rb') as fid:
            video = np.fromfile(fid, dtype=code_format)
        # order the video as image after image
        video = np.reshape(video, (self.video_stats[1][0],
                                   self.video_stats[1][1],
                                   self.video_stats[1][2]), order='F')


        return np.swapaxes(video, 0, 1)

    def videoCut(self):
        """Cut out the area with the grating.

        Modifies the loaded video data stored in object's video attribute.
        """
        while True:
            corners = self.select_image_area(self.video[:, :, 21])
            top_line = int(round((corners[0][1] + corners[2][1])/2))
            bot_line = int(round((corners[1][1] + corners[3][1])/2))+1
            left_line = int(round((corners[0][0] + corners[1][0])/2))
            right_line = int(round((corners[2][0] + corners[3][0])/2))
            new_frame = self.video[top_line:bot_line,
                                   left_line:right_line, 21]
            frame2 = self.frame(new_frame)
            print('Continue with clicking the image.')
            plt.ginput(1)
            plt.close(frame2)

            answer = input('Repeat area inputs [y/n]? ')
            if answer == 'n':
                break
        plt.close()
        self.video = self.video[top_line:bot_line, left_line:right_line, :]
        self.video_stats = (self.video_stats[0],
                            (bot_line - top_line,
                             right_line - left_line,
                             self.video_stats[1][2]))

    def select_image_area(self, image):
        """Select rectangle area of interest from the image.

        Plots an image where the user inputs four points to locate corners of
        the desired rectangle area.

        Return:
        -------
            corners - list of tuples with coordinates of the input points
                      (ordered from the left top corner counter clockwise)
        """
        frame1 = self.frame(image)
        print('Please select the corners of the grating.')
        corners = plt.ginput(4)
        plt.close(frame1)
        return corners

    @staticmethod
    def sort_image_points(points):
#        TODO
        pass


    def frame(self, arg=1, rng=[-1, 1]):

        if type(arg) == int:
            data = self.video[self.view[1]:self.view[1]+self.view[3], self.view[0]:self.view[0]+self.view[2], arg]
        else:
            data = np.array(arg)
        fig, ax = plt.subplots()
        ax.imshow(data, cmap='gray', vmin=rng[0], vmax=rng[1])
        ax.set(xlabel='x [px]', ylabel='y [px]')
        return fig

    def play(self, fr=[0, -1], delta=0.2, rng=[-1, 1]):

        fig, ax = plt.subplots()

        ax.set(xlabel='x [px]', ylabel='y [px]')

        for i in range(fr[0], fr[1]):
            data = self.video[self.view[1]:self.view[1]+self.view[3], self.view[0]:self.view[0]+self.view[2], i]
            ax.set_title(str(i)+'/'+str(fr[1]-fr[0]))
            ax.imshow(data, cmap='gray', vmin=rng[0], vmax=rng[1])
            plt.pause(delta)
        return fig

    @staticmethod
    def show(img, rng):
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray', vmin=rng[0], vmax=rng[1])
        ax.set(xlabel='x [px]', ylabel='y [px]')
        return fig

    def show_area(self, area, rng):
        """shows the selected area

        Parameters
        ----------
        area : list,
            coordinates of the left top point (x, y) and width and height
            [x, y, width, height]
        """

        x, y, width, height = area
        img=self.video[:, :, 1]
        img[y:y + height, x:x + width]=1
        self.show(img, rng)
    def select_area(self, area):
        """reduces the image to the selected area

        Parameters
        ----------
        area : list,
            coordinates of the left top point (x, y) and width and height
            [x, y, width, height]
        """

        x, y, width, height = area
        self.video=self.video[y:y + height, x:x + width, :]
        self.view=[0, 0, width, height]


