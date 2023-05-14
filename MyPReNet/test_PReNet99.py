import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from networks import *
import time 

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="logs/PReNet6/", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="/media/r/BC580A85580A3F20/dataset/rain/peku/Rain100H/rainy", help='path to training data')
parser.add_argument("--save_path", type=str, default="/home/r/works/derain_arxiv/release/results/PReNet", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():

    os.makedirs(opt.save_path, exist_ok=True)
    cleanPath = 'datasets/test/clean306/'
    nocleanPath = 'datasets/test/noclean306/'
    
    # Build model
    print('Loading model ...\n')
    model = PReNet(opt.recurrent_iter, opt.use_GPU)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    for count in range(1,100):
        sum=0.0
#        print("net_epoch"+str(count)+".pth")
        model.load_state_dict(torch.load(os.path.join(opt.logdir, "net_epoch"+str(count)+".pth")))
        model.eval()

        
        for img_name in os.listdir(cleanPath):
            if is_image(img_name):
                img_path_clean = os.path.join(cleanPath, img_name)
                img_path_noclean = os.path.join(nocleanPath, 'no'+img_name)

                # input image
                y1 = cv2.imread(img_path_noclean)
                b1, g1, r1 = cv2.split(y1)
                y1 = cv2.merge([r1, g1, b1])

                y1 = normalize(np.float32(y1))
                y1 = np.expand_dims(y1.transpose(2, 0, 1), 0)
                y1 = Variable(torch.Tensor(y1))

                # input image
                y2 = cv2.imread(img_path_noclean)
                b2, g2, r2 = cv2.split(y2)
                y2 = cv2.merge([r2, g2, b2])

                y2 = normalize(np.float32(y2))
                y2 = np.expand_dims(y2.transpose(2, 0, 1), 0)
                y2 = Variable(torch.Tensor(y2))

                if opt.use_GPU:
                    y = y1.cuda()
                    y2 = y2.cuda()

                with torch.no_grad(): #
                    if opt.use_GPU:
                        torch.cuda.synchronize()
                    #import pdb;pdb.set_trace()
                    out, _ = model(y)
                    out = torch.clamp(out, 0., 1.)

                    if opt.use_GPU:
                        torch.cuda.synchronize()
                    psnr_train = batch_PSNR(out, y2, 1.)
                    sum = sum+psnr_train
#                    print(psnr_train)
        print(sum/306)
                



if __name__ == "__main__":
    main()

