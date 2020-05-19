import urllib
import cv2
import numpy as np
import os
import imutils
import collections
from matplotlib import pyplot as plt


class Plates:
    def __init__(self, ori_img, img_name,pm_thresh, temp_num):
        # ori_img is original plat area

        self.img_name = img_name
        self.ori = ori_img.copy()
        self.result = ori_img
        self.img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
        self.pm_thresh = pm_thresh
        self.temp_num = temp_num
        self.digits = []

        # Do the work
        self.plate_segmentation()
        self.pattern_matching()
        self.show_plt()

    def pattern_matching(self):
        self.pm = []
        _, mask = cv2.threshold(self.img, thresh=200, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.grayscale = cv2.cvtColor(self.ori, cv2.COLOR_BGR2GRAY)
        self.binary = cv2.bitwise_and(self.grayscale, mask)
        method = cv2.TM_CCORR_NORMED
        threshold = self.pm_thresh
        cw, ch = self.binary.shape[::-1]
        self.crop_inv = cv2.bitwise_not(self.binary)
        self.crop_ev = imutils.resize(self.crop_inv, height=120)

        # cv2.imshow("crop", self.crop)
        tmp_result = []
        for digit in self.digits:
            max_sim = 0
            for temp in self.temp_num:
                highest = 0
                highest_pt = []
                # for i in range(1, 4):
                temp_result = []
                # resp = urllib.request.urlopen("https://github.com/keynekassapa13/num-plate-track/blob/master/temp-num/{}-0{}.png?raw=true".format(temp, str(i)))
                # image = np.asarray(bytearray(resp.read()), dtype="uint8")
                # t_img = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                t_img = cv2.imread("./temp-num/{}".format(temp), cv2.IMREAD_GRAYSCALE)
                t_img = cv2.resize(t_img, (98, 98))
                t_img = 255 - t_img
                # t_img = imutils.resize(t_img, height = ch-2)
                w, h = t_img.shape[::-1]

                res = cv2.matchTemplate(digit.astype(np.uint8), t_img.astype(np.uint8), method)
                # loc = np.where( res >= threshold)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                print(max_val)
                if max_val > threshold and max_val > max_sim:
                    max_sim = max_val
                    match_name = temp
            if max_sim > 0:
                self.pm.append(match_name)

        # self.pm = collections.OrderedDict(sorted(self.pm))
        self.pm_result = ''
        for pm in self.pm:
            self.pm_result += pm.split('-')[0]

        print("::::::RESULT = {}\n".format(self.pm_result))
        return

    def square(self, img):
        """
        This function resize non square image to square one (height == width)
        :param img: input image as numpy array
        :return: numpy array
        """

        # image after making height equal to width
        squared_image = img

        # Get image height and width
        h = img.shape[0]
        w = img.shape[1]

        # In case height superior than width
        if h > w:
            diff = h - w
            if diff % 2 == 0:
                x1 = np.zeros(shape=(h, diff // 2))
                x2 = x1
            else:
                x1 = np.zeros(shape=(h, diff // 2))
                x2 = np.zeros(shape=(h, (diff // 2) + 1))

            squared_image = np.concatenate((x1, img, x2), axis=1)

        # In case height inferior than width
        if h < w:
            diff = w - h
            if diff % 2 == 0:
                x1 = np.zeros(shape=(diff // 2, w))
                x2 = x1
            else:
                x1 = np.zeros(shape=(diff // 2, w))
                x2 = np.zeros(shape=((diff // 2) + 1, w))

            squared_image = np.concatenate((x1, img, x2), axis=0)

        return squared_image

    def plate_segmentation(self):

        img = self.ori.copy()
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgray = cv2.bitwise_not(imgray)

        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        scale1 = 0.01
        scale2 = 0.3
        area_condition1 = area * scale1
        area_condition2 = area * scale2
        # global thresholding
        ret1, th1 = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

        # Otsu's thresholding
        ret2, th2 = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(imgray, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        self.digits = []
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            area = w*h
            if area < 200:
                continue
            # if (w * h > area_condition1 and w * h < area_condition2 and w/h > 0.2 and h/w > 1):
            cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            c = th2[y:y + h, x:x + w]
            c = np.array(c)
            c = cv2.bitwise_not(c)
            c = self.square(c)
            c = cv2.resize(c, (98, 98), interpolation=cv2.INTER_AREA)
            self.digits.append(c)
        self.segmented = img

    def show_plt(self):
        title = [
            'Black and White',
            # 'Predicted Num :\n'+ self.pm_result
        ]
        result = [self.ori]
        num = [231, 232, 233, 234, 235]

        for i in range(len(result)):
            plt.subplot(num[i]), plt.imshow(result[i], cmap='gray')
            plt.title(title[i]), plt.xticks([]), plt.yticks([])

        plt.suptitle(self.img_name)
        plt.show()

        plt.imshow(self.segmented, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

        for j in range(len(self.digits)):
            plt.figure(figsize=(1, 1))
            plt.imshow(self.digits[j], cmap='gray')
            plt.xticks([]), plt.yticks([])
        plt.show()