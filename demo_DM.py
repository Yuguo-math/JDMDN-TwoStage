import os
import torch
import numpy as np
import skimage.io as skio
from modules_f import Dcnn_Pytorch
from bayer import bayer
from GBTF import gbtf
import argparse


def main(args):
    device = torch.device("cuda:0")
    print(device)

    DM_net = Dcnn_Pytorch(init_weights=False, mode=args.mode, stage='DM') # 'normal' 'lightweight'
    DM_net.to(device)
    DM_net.eval()

    DM_path_checkpoint = './Demosaic/'+ args.mode + '.pkl'  # 'normal' 'lightweight'
    DM_params_dict = torch.load(DM_path_checkpoint)
    DM_net.load_state_dict(DM_params_dict)        # load net

    _ = []

    data = ['Kodak']

    for j in range(len(data)):
        imgs = os.listdir('./testimage/{}Orginal'.format(data[j]))
        imgs = list(filter(lambda x: x.endswith('.png'), imgs))

        for i in range(len(imgs)):
            im = skio.imread(
                os.path.join('./testimage/{}Orginal'.format(data[j]), imgs[i]))  # h,w,c
            im = np.transpose(im, [2, 0, 1])
            im = im.astype(np.float32) / (2 ** 8 - 1)

            mosaic, imask, mask = bayer(im, True)
            pre_im = gbtf(mosaic.astype('float32'))

            mosaic = mosaic.astype('float32')
            imask = imask.astype('float32')
            pre_im = pre_im.astype('float32')

            pre_im = torch.from_numpy(pre_im).unsqueeze(0).to(device)
            imask = torch.from_numpy(imask).unsqueeze(0).to(device)
            mosaic = torch.from_numpy(mosaic).unsqueeze(0).to(device)

            with torch.no_grad():
                DM_out,_ = DM_net(pre_im,_,_)
                DM_out = DM_out * imask + mosaic

            DM_out = np.transpose((np.clip(DM_out.cpu().numpy().squeeze(0), 0, 1) * 255).astype('uint8'), [1, 2, 0])

            skio.imsave(
                os.path.join('./testimage/result'.format(data[j]),imgs[i]), DM_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='normal',help="Choose normal or lightweight")
    args = parser.parse_args()

    main(args)