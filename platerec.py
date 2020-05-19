import cv2
import numpy as np
import os
import imutils
import collections
from matplotlib import pyplot as plt
import glob
from transform import four_point_transform

class Plates:
    STD_HEIGHT = 200

    def __init__(self, img_ori, img_name):
        self.img_ori = self.preprocess(img_ori)
        self.result = self.img_ori.copy()
        self.img_name = img_name
        self.img = cv2.cvtColor(self.img_ori, cv2.COLOR_BGR2GRAY)

        self.img_process()
        self.contour()
        self.show_plt()

    def preprocess(self, img_ori):
        self.height = img_ori.shape[0]
        self.width = img_ori.shape[1]

        # Resize image
        self.width = int(self.STD_HEIGHT / self.height * self.width)
        self.height = self.STD_HEIGHT
        img = cv2.resize(img_ori, (self.width, self.height))
        return img

    def img_process(self):
        # Threshold
        _, mask = cv2.threshold(self.img, thresh=200, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        self.res_img = mask

        # Morphological Transformation
        kernel = np.ones((3, 3), np.uint8)
        # self.res_img = cv2.erode(self.res_img, kernel=kernel, iterations=1)
        # self.res_img = cv2.dilate(self.res_img, kernel=kernel, iterations=1)
        # self.res_img = cv2.dilate(self.res_img, kernel=kernel, iterations=1)
        # self.res_img = cv2.erode(self.res_img, kernel=kernel, iterations=1)

        self.edges = cv2.Canny(self.res_img, self.height, self.width)

    def contour(self):
        # Contours
        contours, _ = cv2.findContours(
            self.res_img,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        NumberPlateCnt = None
        found = False
        lt, rb = [10000, 10000], [0, 0]

        # Calculate polygonal curve, see if it has 4 curve
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)
            if len(approx) == 4:
                found = True
                NumberPlateCnt = approx
                break
        if found:
            cv2.drawContours(self.result, [NumberPlateCnt], -1, (255, 0, 255), 2)

            for point in NumberPlateCnt:
                cur_cx, cur_cy = point[0][0], point[0][1]
                if cur_cx < lt[0]: lt[0] = cur_cx
                if cur_cx > rb[0]: rb[0] = cur_cx
                if cur_cy < lt[1]: lt[1] = cur_cy
                if cur_cy > rb[1]: rb[1] = cur_cy

            cv2.circle(self.result, (lt[0], lt[1]), 2, (150, 200, 255), 2)
            cv2.circle(self.result, (rb[0], rb[1]), 2, (150, 200, 255), 2)

            self.crop = self.res_img[lt[1]:rb[1], lt[0]:rb[0]]
            self.crop_res = self.img_ori[lt[1]:rb[1], lt[0]:rb[0]]

    def pattern_matching(self):
        self.pm = {}

        method = cv2.TM_CCOEFF_NORMED
        threshold = self.pm_thresh
        cw, ch = self.crop.shape[::-1]

        # cv2.imshow("crop", self.crop)

        for temp in self.temp_num:
            highest = 0
            highest_pt = []
            for i in range(1, 4):
                temp_result = []
                t_img = cv2.imread("./temp-num/{}-0{}.png".format(temp, str(i)), 0)
                t_img = imutils.resize(t_img, height=ch - 2)
                w, h = t_img.shape[::-1]

                res = cv2.matchTemplate(self.crop, t_img, method)
                loc = np.where(res >= threshold)
                for pt in zip(*loc[::-1]):
                    temp_result.append(pt)
                    cv2.rectangle(self.crop_res, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
                if len(temp_result) > highest:
                    highest = len(temp_result)
                    highest_pt = temp_result

            for pt in highest_pt:
                cv2.rectangle(self.crop_res, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
                self.pm[pt[0]] = temp

        self.pm = collections.OrderedDict(sorted(self.pm.items()))
        self.pm_result = ''
        for _, pm in self.pm.items():
            self.pm_result += pm

        print("::::::RESULT = {}\n".format(self.pm_result))
        return

    def show_plt(self):
        '''
        Showing 6 main step of the process for turning original frame to the result frame
        '''
        title = [
            'Black and White',
            'Threshold',
            'Canny',
            'Num Plate Detected',
            'Num Plate Cropped',
            'Predicted Num :\n'#+ self.pm_result
        ]
        result = [self.img, self.res_img, self.edges, self.result[:, :, ::-1], self.crop, self.crop_res]
        num = [231, 232, 233, 234, 235, 236]

        for i in range(len(result)):
            plt.subplot(num[i]),plt.imshow(result[i], cmap = 'gray')
            plt.title(title[i]), plt.xticks([]), plt.yticks([])

        plt.suptitle(self.img_name)
        plt.show()

if __name__ == '__main__':
    files = glob.glob('plat/*')
    for f in files:
        ori_img = cv2.imread(f)
        Plates(ori_img, f)
