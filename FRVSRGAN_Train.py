"""
This file trains a FRVSR-GAN model on based on an upscaling factor of 4x.
Aman Chadha | aman@amanchadha.com

Adapted from FR-SRGAN, MIT 6.819 Advances in Computer Vision, Nov 2018
"""

import argparse
import os
from math import log10
import gc

import numpy as np
import pandas as pd
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import DatasetLoader
import logger
from FRVSRGAN_Models import FRVSR
from FRVSRGAN_Models import GeneratorLoss
from SRGAN.model import Discriminator
import SRGAN.pytorch_ssim as pts
from visdom import Visdom
from torch import tensor as tt

################################################## iSEEBETTER TRAINER KNOBS #############################################
UPSCALE_FACTOR = 4
########################################################################################################################

# Handle command line arguments
parser = argparse.ArgumentParser(description='Train iSeeBetter: Super Resolution Models')
parser.add_argument('-o', '--out_path', default='.', type=str, help='output path')
parser.add_argument('-e', '--num_epochs', default=1000, type=int, help='train epoch number')
parser.add_argument('-w', '--width', default=64, type=int, help='lr pic width')
parser.add_argument('-ht', '--height', default=64, type=int, help='lr pic height')
parser.add_argument('-d', '--dataset_size', default=0, type=int, help='dataset_size, 0 to use all')
parser.add_argument('-b', '--batchSize', default=2, type=int, help='batchSize, default 2')
parser.add_argument('-l', '--lr', default=1e-5, type=float, help='learning rate, default 1e-5')
parser.add_argument('-x', '--express', default=False, action='store_true', help='Express mode: no validation.')
parser.add_argument('-v', '--debug', default=False, action='store_true', help='Print debug spew.')
parser.add_argument('--visdom_host', default='localhost', type=str, help='visdom host')

args = parser.parse_args()
OUT_PATH = args.out_path
os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(f'{OUT_PATH}/epochs', exist_ok=True)
os.makedirs(f'{OUT_PATH}/statistics', exist_ok=True)

# visualize every N iterations
visdom_iter = 20
# save model every N iterations
save_iter = 1000

NUM_EPOCHS = args.num_epochs
WIDTH = args.width
HEIGHT = args.height
batchSize = args.batchSize
dataset_size = args.dataset_size
lr = args.lr
express = args.express

# Load dataset
trainLoader, valLoader = DatasetLoader.get_data_loaders(batchSize, dataset_size=dataset_size, validation_split=0.1)
numTrainBatches = len(trainLoader)
numValBatches = len(valLoader)

# Initialize Logger
logger.initLogger(args.debug)

# Use Generator as FRVSR
netG = FRVSR(batchSize, lr_width=WIDTH, lr_height=HEIGHT)
print('# of Generator parameters:', sum(param.numel() for param in netG.parameters()))

# Use Discriminator from SRGAN
netD = Discriminator()
print('# of Discriminator parameters:', sum(param.numel() for param in netD.parameters()))

generatorCriterion = GeneratorLoss()

if torch.cuda.is_available():
    def printCUDAStats():
        logger.info("# of CUDA devices detected: %s", torch.cuda.device_count())
        logger.info("Using CUDA device #: %s", torch.cuda.current_device())
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(torch.cuda.current_device()))


    printCUDAStats()

    netG.cuda()
    netD.cuda()
    generatorCriterion.cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Use Adam optimizer
optimizerG = optim.Adam(netG.parameters(), lr=lr)
optimizerD = optim.Adam(netD.parameters(), lr=lr)

vis = Visdom(port=8097, server=args.visdom_host)
gt_patch_size = HEIGHT * UPSCALE_FACTOR
lr_win = vis.images(np.random.randn(batchSize, 3, gt_patch_size, gt_patch_size), opts=dict(title='lr'))
hr_win = vis.images(np.random.randn(batchSize, 3, gt_patch_size, gt_patch_size), opts=dict(title='hr'))
sr_win = vis.images(np.random.randn(batchSize, 3, gt_patch_size, gt_patch_size), opts=dict(title='sr'))
ZERO = torch.zeros(1).cpu()
dloss_opts = dict(xlabel='minibatches', ylabel='loss', title='Dloss', legend=['Dloss'])
gloss_opts = dict(xlabel='minibatches', ylabel='loss', title='Gloss', legend=['Gloss'])
dscore_opts = dict(xlabel='minibatches', ylabel='loss', title='DScore', legend=['D(x)'])
gscore_opts = dict(xlabel='minibatches', ylabel='loss', title='GScore', legend=['D(G(z))'])
dloss_win = vis.line(X=ZERO, Y=ZERO, opts=dloss_opts)
gloss_win = vis.line(X=ZERO, Y=ZERO, opts=gloss_opts)
dscore_win = vis.line(X=ZERO, Y=ZERO, opts=dscore_opts)
gscore_win = vis.line(X=ZERO, Y=ZERO, opts=gscore_opts)

UP = torch.nn.Upsample(mode='bilinear', scale_factor=UPSCALE_FACTOR)


def trainModel(epoch):
    trainBar = tqdm(trainLoader)
    runningResults = {'batchSize': 0, 'DLoss': 0, 'GLoss': 0, 'DScore': 0, 'GScore': 0}

    netG.train()
    netD.train()

    for data, target in trainBar:
        batchSize = data.size(0)
        runningResults['batchSize'] += batchSize

        ################################################################################################################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ################################################################################################################
        fakeHRs = []
        fakeLRs = []
        fakeScrs = []
        realScrs = []
        DLoss = 0
        LRImgs = []
        HRImgs = []

        # Zero-out gradients, i.e., start afresh
        netD.zero_grad()

        netG.init_hidden(device)

        for LRImg, HRImg in zip(data, target):
            HRImg = HRImg.to(device)
            LRImg = LRImg.to(device)

            fakeHR, fakeLR = netG(LRImg)

            realOut = netD(HRImg).mean()
            fake_out = netD(fakeHR).mean()

            fakeHRs.append(fakeHR)
            fakeLRs.append(fakeLR)
            LRImgs.append(LRImg)
            HRImgs.append(HRImg)
            fakeScrs.append(fake_out)
            realScrs.append(realOut)

            DLoss += 1 - realOut + fake_out

        DLoss /= len(data)

        # Calculate gradients
        DLoss.backward(retain_graph=True)

        # Update weights
        optimizerD.step()

        ################################################################################################################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ################################################################################################################
        GLoss = 0

        # Zero-out gradients, i.e., start afresh
        netG.zero_grad()

        idx = 0
        for fakeHR, fakeLR, fake_scr, HRImg, LRImg in zip(fakeHRs, fakeLRs, fakeScrs, target, data):
            fakeHR = fakeHR.to(device)
            fakeLR = fakeLR.to(device)
            fake_scr = fake_scr.to(device)
            HRImg = HRImg.to(device)
            LRImg = LRImg.to(device)
            GLoss += generatorCriterion(fake_scr, fakeHR, HRImg, fakeLR, LRImg, idx)
            idx += 1

        GLoss /= len(data)

        # Calculate gradients
        GLoss.backward()

        # Update weights
        optimizerG.step()

        realOut = torch.Tensor(realScrs).mean()
        fake_out = torch.Tensor(fakeScrs).mean()
        runningResults['GLoss'] += GLoss.data.item() * batchSize
        runningResults['DLoss'] += DLoss.data.item() * batchSize
        runningResults['DScore'] += realOut.data.item() * batchSize
        runningResults['GScore'] += fake_out.data.item() * batchSize

        dloss = runningResults['DLoss'] / runningResults['batchSize']
        gloss = runningResults['GLoss'] / runningResults['batchSize']
        dscore = runningResults['DScore'] / runningResults['batchSize']
        gscore = runningResults['GScore'] / runningResults['batchSize']
        trainBar.set_description(desc='[Epoch: %d/%d] D Loss: %.4f G Loss: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, dloss, gloss, dscore, gscore))

        current_step = trainBar.n
        if current_step % visdom_iter == 0:
            lrimg = LRImgs[len(LRImgs) // 2]
            hrimg = HRImgs[len(HRImgs) // 2]
            srimg = fakeHRs[len(fakeHRs) // 2]
            vis.images(torch.clamp(UP(lrimg), 0, 1), opts=dict(title='LR'), win=lr_win)
            vis.images(srimg, opts=dict(title='SR'), win=sr_win)
            vis.images(hrimg, opts=dict(title='HR'), win=hr_win)
            vis.line(X=tt([current_step]), Y=tt([dloss]).cpu(), win=dloss_win, update='append', opts=dloss_opts)
            vis.line(X=tt([current_step]), Y=tt([gloss]).cpu(), win=gloss_win, update='append', opts=gloss_opts)
            vis.line(X=tt([current_step]), Y=tt([dscore]).cpu(), win=dscore_win, update='append', opts=dscore_opts)
            vis.line(X=tt([current_step]), Y=tt([gscore]).cpu(), win=gscore_win, update='append', opts=gscore_opts)

        if current_step != 0 and current_step % save_iter == 0:
            saveModelParams(epoch, runningResults, iter=current_step)

        gc.collect()

    netG.eval()

    return runningResults

def validateModel():
    validationBar = tqdm(valLoader)
    validationResults = {'MSE': 0, 'SSIMs': 0, 'PSNR': 0, 'SSIM': 0, 'batchSize': 0}
    for valLR, valHR in validationBar:
        batchSize = valLR.size(0)
        validationResults['batchSize'] += batchSize

        netG.init_hidden(device)

        batchMSE = []
        batchSSIM = []
        for lr, hr in zip(valLR, valHR):
            lr = lr.to(device)
            hr = hr.to(device)

            HREst, LREst = netG(lr)
            batchMSE.append(((HREst - hr) ** 2).data.mean())
            batchSSIM.append(pts.ssim(HREst, hr).item())

        batchMSE = torch.Tensor(batchMSE).mean()
        validationResults['MSE'] += batchMSE * batchSize
        batchSSIM = torch.Tensor(batchSSIM).mean()
        validationResults['SSIMs'] += batchSSIM * batchSize
        validationResults['PSNR'] = 10 * log10(1 / (validationResults['MSE'] / validationResults['batchSize']))
        validationResults['SSIM'] = validationResults['SSIMs'] / validationResults['batchSize']
        validationBar.set_description(desc='[Converting LR images to SR images] PSNR: %.4fdB SSIM: %.4f' %
                                      (validationResults['PSNR'], validationResults['SSIM']))
        gc.collect()

        return validationResults

def saveModelParams(epoch, runningResults, iter, validationResults={}):
    results = {'DLoss': [], 'GLoss': [], 'DScore': [], 'GScore': [], 'PSNR': [], 'SSIM': []}

    # Save model parameters
    if iter is not None:
        torch.save(netG.state_dict(), f'{OUT_PATH}/epochs/netG_epoch_{UPSCALE_FACTOR}_{epoch}_{iter}.pth')
        torch.save(netG.state_dict(), f'{OUT_PATH}/epochs/netD_epoch_{UPSCALE_FACTOR}_{epoch}_{iter}.pth')
    else:
        torch.save(netG.state_dict(), '%s/epochs/netG_epoch_%d_%d.pth' % (OUT_PATH, UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), '%s/epochs/netD_epoch_%d_%d.pth' % (OUT_PATH, UPSCALE_FACTOR, epoch))

    # Save Loss\Scores\PSNR\SSIM
    results['DLoss'].append(runningResults['DLoss'] / runningResults['batchSize'])
    results['GLoss'].append(runningResults['GLoss'] / runningResults['batchSize'])
    results['DScore'].append(runningResults['DScore'] / runningResults['batchSize'])
    results['GScore'].append(runningResults['GScore'] / runningResults['batchSize'])
    #results['PSNR'].append(validationResults['PSNR'])
    #results['SSIM'].append(validationResults['SSIM'])

    if epoch % 1 == 0 and epoch != 0 and iter is None:
        out_path = f'{OUT_PATH}/statistics/'
        data_frame = pd.DataFrame(data={'DLoss': results['DLoss'], 'GLoss': results['GLoss'], 'DScore': results['DScore'],
                                  'GScore': results['GScore']},#, 'PSNR': results['PSNR'], 'SSIM': results['SSIM']},
                                  index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'FRVSRGAN__' + str(UPSCALE_FACTOR) + '_Train_Results.csv', index_label='Epoch')

def main():
    """ Lets begin the training process! """

    for epoch in range(1, NUM_EPOCHS + 1):
        runningResults = trainModel(epoch)

        # Do validation only if express mode is not enabled
        if not express:
            validationResults = validateModel()

        saveModelParams(epoch, runningResults)#, validationResults)

if __name__ == "__main__":
    main()
