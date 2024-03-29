import torch
import cv2
import os
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import OrderedDict
from saleval import SalEval
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
@torch.no_grad()
def test(args, model, image_list, label_list, save_dir):
    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]
    eval = SalEval()
    for idx in tqdm(range(len(image_list))):
        image = cv2.imread(image_list[idx])
        label = cv2.imread(label_list[idx], 0)
        label = label / 255.0
        # resize the image to 1024x512x3 as in previous papers
        img = cv2.resize(image, (args.width, args.height))
        img = img.astype(np.float32) / 255.0
        img -= mean
        img /= std
        img = img[:,:, ::-1].copy()
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        img = Variable(img)
        label = torch.from_numpy(label).float().unsqueeze(0)
        if args.gpu:
            img = img.cuda()
            label = label.cuda()
        shape = image.shape[:2]
        img_out = model(img,shape)[:, 0, :, :].unsqueeze(dim=0)
        eval.add_batch((img_out[:, 0, :, :]), label.unsqueeze(dim=0))
        sal_map = np.round((img_out[0,0]*255).data.cpu().numpy())
        cv2.imwrite(osp.join(save_dir, osp.basename(image_list[idx])[:-4] + '.png'), sal_map)

def main(args, file_list):
    # read all the images in the folder
    image_list = list()
    label_list = list()
    with open(args.data_dir + '/' + file_list + '.txt') as fid:
        for line in fid:
            line_arr = line.split()
            image_list.append(args.data_dir + '/' + line_arr[0].strip())
            label_list.append(args.data_dir + '/' + line_arr[1].strip())


    model = torch.load("R50.pth")

    if args.gpu:
        model = model.cuda()
    # set to evaluation mode
    model.eval()
    save_dir = args.savedir + '/' + file_list + '/'
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    test(args, model, image_list, label_list, save_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./data", help='Data directory')
    parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
    parser.add_argument('--savedir', default='./salmaps/R50Sal/', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),help='Run on CPU or GPU. If TRUE, then GPU')
    args = parser.parse_args()
    print('Called with args:')
    print(args)
    data_lists = ['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'PASCAL-S', 'HKU-IS',]
    for data_list in data_lists:
        print("processing ", data_list)
        main(args, data_list)
