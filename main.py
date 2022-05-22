""" a modified version of deep-text-recognition-benchmark repository https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/test.py """

from multiprocessing.sharedctypes import Value
import os
import time
import string
import argparse
import re
import cv2
import imutils
from datetime import datetime
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

import sys
sys.path.insert(0, 'AttentionHTR/model/')
from AttentionHTR.model.utils import AttnLabelConverter
from AttentionHTR.model.model import Model
from word_segmentation import Word
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HTR():
    def __init__(self, opt, model, converter):
        self.opt = opt
        self.model = model
        self.converter = converter
        self.preds = []
        self.borders = []
        self.paths = []

    def _resize(self, thresh, width, height):
        (tH, tW) = thresh.shape
        # if the width is greater than the height, resize along the
        # width dimension
        if tW > tH:
            thresh = imutils.resize(thresh, width=width)
        # otherwise, resize along the height
        else:
            thresh = imutils.resize(thresh, height=height)

        # re-grab the image dimensions (now that its been resized)
        # and then determine how much we need to pad the width and
        # height such that our image will be 32x32
        (tH, tW) = thresh.shape
        dX = int(max(0, width - tW) / 2.0)
        dY = int(max(0, height - tH) / 2.0)
        # pad the image and force widthxheight dimensions
        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
            left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255))
        padded = cv2.resize(padded, (width, height))
        return padded

    def _preprocess(self, img):
        # black and white
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # remove noise
        se=cv2.getStructuringElement(cv2.MORPH_RECT, (30,30))
        bg=cv2.morphologyEx(gray, cv2.MORPH_DILATE, se)
        out_gray=cv2.divide(gray, bg, scale=255)
        
        # resize image to fit model required size (32x100)
        img = self._resize(out_gray, opt.imgW, opt.imgH)
        img = img/255.0
        img = np.expand_dims(img, axis=0)
        return img

    def _make_tensor(self, words):
        # convert all words to one tensor
        imgs = [self._preprocess(img) for img in words]
        imgs = np.stack(imgs)
        imgs = torch.tensor(imgs).float()
        return imgs

    def retrieve_images(self):
        # find image or images names in given path
        path = self.opt.image
        valid_images = [".jpg", ".png",".jpeg"]
        imgs = []

        # path contains directory, find all images in directory
        if path[-1] == '/':
            for f in sorted(os.listdir(path)):
                ext = os.path.splitext(f)[1]
                if ext.lower() not in valid_images and os.path.exists(os.path.join(path,f)):
                    continue
                imgs.append(os.path.join(path,f))
        
        # path contains image, find image
        else:
            ext = os.path.splitext(path)[1]
            if ext.lower() in valid_images and os.path.exists(path):
                imgs.append(path)
        
        # no image could be found
        if len(imgs) == 0:
            raise Exception('Could not find images!')
        
        self.paths = imgs

    def predict(self, image_tensor):
        # predict words for tensor
        model = self.model
        converter = self.converter
        opt = self.opt

        batch_size = image_tensor.size(0)
        image = image_tensor.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        start_time = time.time()
        preds = model(image, text_for_pred, is_train=False)
        forward_time = time.time() - start_time

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

        self.infer_time += forward_time

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        final_preds = []
        for pred, pred_max_prob in zip( preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)

            final_preds.append(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = float(pred_max_prob.cumprod(dim=0)[-1].cpu().numpy())
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
        
        result = {'final_preds': final_preds,
                  'confidence_score_list': confidence_score_list,
                  'forward_time': forward_time,
        }
        return result

    def generate_output(self, res):
        # return output to stdout
        print()
        print(f"prediction:\t\t{' '.join(res['final_preds'])}")
        print()
        print(f"average confidence:\t{np.mean(res['confidence_score_list'])*100:.2f}%")
        print(f"runtime:\t\t{res['forward_time']:.2f}s")
        print()
        print(f"predicted words:\t{res['final_preds']}\n")

    def create_images(self):
        # generate images with predictions
        figs = []
        for i, fp in enumerate(self.paths):
            img = cv2.imread(fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fig = self.make_image(img, self.borders[i], self.preds[i])
            figs.append(fig)
        return figs

    def make_image(self, image, boxes, preds):
        fig, ax = plt.subplots( nrows=1, ncols=1)
    
        # loop over the predictions and bounding box locations together
        for (idx, (x, y, xw, yh)) in enumerate(boxes):

            # draw the prediction on the image
            cv2.rectangle(image, (x, y), (xw, yh), (0, 255, 0), 3)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            text = cv2.putText(image, preds[idx], (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # show the image
            ax.imshow(img)
            ax.imshow(text)
        return fig

    def save_image(self, i, fp, fig):
        # create unique name in output directory
        date = datetime.now().strftime("%d%m%Y_%H%M%S")
        fp, ext = os.path.splitext(fp)
        fn = fp.rsplit('/', 1)[-1]
        fn = f"{fn}_ocr_{date}"
        save_path = f"{self.opt.output}/{fn}{ext}"

        # save image to output directory
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f'{i}\tSuccesfully saved image to {save_path}!')
    
    def save_images(self, figs):
        # create output directory if not existent
        if not os.path.exists(self.opt.output):
            os.makedirs(self.opt.output)

        # save all found images
        for i, (fp, fig) in enumerate(zip(self.paths, figs)):
            self.save_image(f"({i+1}/{len(self.paths)})", fp, fig)

    def make_predictions(self):
        """ validation or evaluation """
        self.infer_time = 0
        self.retrieve_images()
        imgs = []
        for fp in self.paths:
            print(f'|-- Preprocessing {fp} --|')
            W = Word(opt=self.opt)
            words, border = W.word_segmentation(fp)
            imgs.append(self._make_tensor(words))
            self.borders.append(border)

        for i, image_tensors in enumerate(imgs):
            analyzing_text = f'|-- Analyzing image {i+1}: {self.paths[i]} --|'
            stripes = '-'*(len(analyzing_text))
            print(f"{stripes}\n{analyzing_text}\n{stripes}")
            res = self.predict(image_tensors)
            self.preds.append(res['final_preds'])
            self.generate_output(res)
        
        print(f"Total runtime:\t\t{self.infer_time:.2f}s\n")
        
        if self.opt.save:
            print('|-- Saving images --|')
            images = self.create_images()
            self.save_images(images)


def main(opt):
    """ model configuration """
    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ evaluation """
    model.eval()
    with torch.no_grad():
        system = HTR(opt=opt, model=model, converter=converter)
        system.make_predictions()
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='path to images')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default='AttentionHTR/model/saved_models/AttentionHTR-General-sensitive.pth', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', type=int, default=1, help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    """ EAST model Architecture """
    parser.add_argument('--east', type=str, default='EAST-Detector-for-text-detection-using-OpenCV/frozen_east_text_detection.pb', help='Path to EAST model')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='EAST model minimal confidence')
    parser.add_argument('--width', type=int, default=320, help='EAST resize width')
    parser.add_argument('--height', type=int, default=320, help='EAST resize height')
    """ Word segmentation """
    parser.add_argument('--line_dilation', type=int, default=3, help='Line dilation y-kernel value')
    parser.add_argument('--word_dilation', type=int, default=3, help='Word dilation y-kernel value')
    parser.add_argument('--aspect_word', type=int, default=30, help='Limit aspect ratio of found word (width/height)')
    """ Save image """
    parser.add_argument('--save', type=int, default=0, help='Save prediction image')
    parser.add_argument('--output', type=str, default='output', help='Output directory for saving images')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    main(opt)
