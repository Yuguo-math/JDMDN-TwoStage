import os
import torch
import numpy as np
import skimage.io as skio
from modules_skip import Dcnn_Pytorch
from bayer import bayer, bayer_to_mosaic
from GBTF import gbtf
import scipy.io as scio
import argparse


def main(args):
    device = torch.device("cuda:0")
    print(device)

    print(args.mode)

    DM_net = Dcnn_Pytorch(init_weights=False, mode=args.mode, stage='DM') # 'normal' 'lightweight'
    DM_net.to(device)
    DM_net.eval()

    DM_path_checkpoint = './Demosaic/'+ args.mode + '.pkl'  # 'normal' 'lightweight'
    DM_params_dict = torch.load(DM_path_checkpoint)
    DM_net.load_state_dict(DM_params_dict)        # load net

    DN_net = Dcnn_Pytorch(init_weights=False, mode=args.mode, stage='DN') # 'normal' 'lightweight'
    DN_net.to(device)
    DN_net.eval()

    DN_path_checkpoint = './Denoise/' + args.mode + '/end_skip_sigma{}.pkl'.format(args.sigma)  # 'normal' 'lightweight'
    DN_params_dict = torch.load(DN_path_checkpoint, map_location={'cuda:5':'cuda:0'})
    DN_net.load_state_dict(DN_params_dict)  # load net

    _ = []

    dataset = ['Kodak']

    for j in range(len(dataset)):
        imgs = os.listdir('./testimage/{}sigma{}'.format(dataset[j],args.sigma))
        imgs = list(filter(lambda x: x.endswith('.mat'), imgs))

        for i in range(len(imgs)):
            data = scio.loadmat('./testimage/{}sigma{}/{}'.format(dataset[j],args.sigma, imgs[i]))
            im = data['DN']

            mosaic, imask = bayer_to_mosaic(im)
            pre_im = gbtf(mosaic.astype('float32'))

            mosaic = mosaic.astype('float32')
            imask = imask.astype('float32')
            pre_im = pre_im.astype('float32')

            pre_im = torch.from_numpy(pre_im).unsqueeze(0).to(device)
            imask = torch.from_numpy(imask).unsqueeze(0).to(device)
            mosaic = torch.from_numpy(mosaic).unsqueeze(0).to(device)

            with torch.no_grad():
                DM_out,skip = DM_net(pre_im,_)
                DM_out = DM_out * imask + mosaic

                DN_out = DN_net(DM_out,skip)

            DN_out = np.transpose((np.clip(DN_out.cpu().numpy().squeeze(0), 0, 1) * 255).astype('uint8'), [1, 2, 0])

            skio.imsave(
                os.path.join('./testimage/result/', imgs[i].replace('.mat','.png') ), DN_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='normal',help="Choose normal or lightweight")
    parser.add_argument("--sigma", type=int, default=20, help="noise sigma, fixed values include 3, 5, 10, 15, 20, 40, and 60")
    args = parser.parse_args()

    main(args)