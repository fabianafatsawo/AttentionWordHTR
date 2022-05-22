import cv2
import numpy as np
from custom_east import EAST_box_detector


class Word():
    def __init__(self, opt):
        self.opt = opt
        self.line_dilation = int(opt.line_dilation)
        self.word_dilation = int(opt.word_dilation)
        self.args = {'east': opt.east,
                     'min_confidence': opt.min_confidence,
                     'width': opt.width,
                     'height': opt.height
        }

    def thresholding(self, image):
        # black and white
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # remove noise
        se=cv2.getStructuringElement(cv2.MORPH_RECT , (30,30))
        bg=cv2.morphologyEx(gray, cv2.MORPH_DILATE, se)
        out_gray=cv2.divide(gray, bg, scale=255)
        
        # remove white pixels
        thresh = cv2.threshold(out_gray, 127, 255, cv2.THRESH_BINARY_INV)[1] 
        return thresh

    def dilation(self, img, kernel, iterations):
        # perform dilation operation
        kernel = np.ones(kernel, np.uint8)
        dilated = cv2.dilate(img, kernel, iterations = iterations)
        return dilated

    def _overlap(self, start1, end1, start2, end2):
        # check if boxes overlap
        return len(set(range(start1, end1+1)).intersection(range(start2, end2+1))) > 0

    def _compare_approved_lines(self, y, h, approved_lines):
        # check if lines and approved lines overlap
        for (mean, delta) in approved_lines:
            if self._overlap(mean-delta, mean+delta, y, y+h):
                return 1
        return 0

    def select_lines(self, cnts, approved_lines):
        # select lines according to approved lines
        approved_cnts = []
        for line in cnts:
            _, y, _, h = cv2.boundingRect(line)

            if self._compare_approved_lines(y, h, approved_lines):
                approved_cnts.append(line)
        return approved_cnts

    def find_words_in_lines(self, sorted_contours_lines, dilated):
        # find coordinates of words in lines
        words_list = []

        for line in sorted_contours_lines:
            # roi of each line
            x, y, w, h = cv2.boundingRect(line)
            h2 = h
            roi_line = dilated[y:y+h, x:x+w]
            flatten = np.average(roi_line, axis=0)
            roi_line = np.expand_dims(flatten, axis=0).astype('uint8')
            
            # draw contours on each word
            cnts = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            sorted_contour_words = sorted(cnts, key=lambda cntr : cv2.boundingRect(cntr)[0])
            
            for word in sorted_contour_words:
                x2, y2, w2, _ = cv2.boundingRect(word)
                aspect_ratio = w2/h2

                # width height ratio limit 
                if aspect_ratio > self.opt.aspect_word:
                    continue
                
                words_list.append([x+x2, y+y2, x+x2+w2, y+y2+h2])
        return words_list

    def word_segmentation(self, fp):
        # find individual words in image
        self.args['image'] = fp
        img = cv2.imread(fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # detect height of lines found by EAST detector
        approved_lines = EAST_box_detector(self.args)

        # remove background
        thresh_img = self.thresholding(img)

        # smear lines
        dilated = self.dilation(thresh_img, kernel=(self.line_dilation,85), iterations=10)
        
        # detect lines
        cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if len(approved_lines):
            cnts = self.select_lines(cnts, approved_lines)
        
        # calculate kernel for word dilation
        avg_line_h = np.mean([cv2.boundingRect(ctr)[3] for ctr in cnts])
        aspect_x = int(avg_line_h*0.55)  # character width by average font aspect value
        sorted_contours_lines = sorted(cnts, key = lambda ctr : cv2.boundingRect(ctr)[1])
        
        # smear words
        dilated = self.dilation(thresh_img, kernel=(self.word_dilation,aspect_x), iterations=1)
        
        # find words in lines
        words_list = self.find_words_in_lines(sorted_contours_lines, dilated)
        
        # crop words out of image
        words_imgs = [img[w[1]:w[3], w[0]:w[2]] for w in words_list]
        return words_imgs, words_list
