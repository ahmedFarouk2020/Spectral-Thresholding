import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



class SpectralThresholding():
    def __init__(self):
        self.__img = np.array([])
        self.__HistValues = None
        self.__peaks = []
        self.__background = 0

    def __convertToGrayscale(self):
        if self.__img.shape.__len__() > 2:
            self.__img = cv.cvtColor(self.__img,cv.COLOR_RGB2GRAY)
        else:
            pass

    def __extractPeaks(self):    
        for i in range(0,len(self.__HistValues)-9):
            if max(list(self.__HistValues[i:i+4])+list(self.__HistValues[i+5:i+9]))<self.__HistValues[i+4]:
                self.__peaks.append((i+4, self.__HistValues[i+4])) # index, value
        
        self.__peaks.append((50, self.__HistValues[-1])) # in histogram scale


    def __getBackgroundLocation(self):
        # we rely on an assumption that background color has max freq in histogram


        # Determine background in the thresholded image to make it black
        # index of background's peak also equals to index of segment in segments list
        self.__background = self.__HistValues.argmax()
        if self.__background <= self.__peaks[0][0]:
            self.__background = 0
        elif self.__background >= self.__peaks[-2][0] or self.__background <= self.__peaks[-1][0]:
            self.__background = len(self.__peaks)-1
        else:
            pass

        # get max peak value -> background
        max_peak_value = max(self.__HistValues)

        # check if background is one of the peaks -> assign it to index of the peak
        for i in range(len(self.__peaks)-1):
            if max_peak_value == self.__peaks[i][1]:
                self.__background = i

    def __getThresholdPoints(self):

        self.__threshold_list = [0]
        for i in range(len(self.__peaks)-1):
            self.__threshold_list.append(((self.__peaks[i][0]*5+2) + \
                (self.__peaks[i+1][0]*5+2))//2)

        self.__threshold_list.append(255)


    def __splitIntoSegments(self):
        self.__segments = []
        for i in range(len(self.__threshold_list)-1):

            self.__segments.append((self.__threshold_list[i] < self.__img) & \
                (self.__img <= self.__threshold_list[i+1]))
    

    def doThresholding(self, img: np.array):
        self.__img = img
        self.__convertToGrayscale()

        self.__HistValues = plt.hist(self.__img.flat, bins=51)[0]
        plt.show()

        self.__HistValues = self.__HistValues.astype(int)

        self.__extractPeaks()
        self.__getBackgroundLocation()
        self.__getThresholdPoints()

        self.__splitIntoSegments()

        # combine all segments to form segmented image
        all_segments = np.zeros((self.__img.shape[0], self.__img.shape[1], 3))
        segments_colors = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(0,1,1),(1,0,1),(1,1,1)]
        for i in range(len(self.__segments)):
            if self.__background == i:
                all_segments[self.__segments[i]] = (0,0,0)
            else:
                all_segments[self.__segments[i]] = segments_colors[i]

        return all_segments

img = cv.imread('images/dots2.jpg', cv.IMREAD_GRAYSCALE)

thresholded_img = SpectralThresholding().doThresholding(img)
plt.imshow(thresholded_img, cmap='gray')
plt.show()






    