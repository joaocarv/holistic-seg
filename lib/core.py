import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from argparse import ArgumentParser

from lib.utils import str2bool, eval_metrics
from lib.sdn import SDN, ResSDN
import time
import os
import csv


class VGGBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch,kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class VGGBlockMixSDN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            SDN(out_ch, out_ch, state_size=150, dirs=[0,2], kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class VGGBlockResMixSDN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            ResSDN(out_ch, out_ch,state_size=150, dirs=[0,2], kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

def return_block(block):
    blocks = dict({'vgg': VGGBlock,
                   'vgg_sdn': VGGBlockMixSDN,
                   'vggres_sdn': VGGBlockResMixSDN})
    block_name = blocks[block]

    return block_name


class Down(nn.Module):
    """
    Downscale with MaxPool => DoubleConvolution block
    """

    def __init__(self, in_ch: int, out_ch: int, block: str):
        super().__init__()
        block_name = return_block(block)
        #print(block_name)
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            block_name(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path, followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int, block: str, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        block_name = return_block(block)

        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = block_name(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ToyNet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, n_layers, lr, feat_start, bilinear, block, sdn_b_enc,sdn_b_dec,
                 sdn_n_layers,loss, viz, n_gpus, exp_name, **kw):
        super(ToyNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.lr = lr
        self.bilinear = bilinear
        self.block = block
        self.sdn_b_enc = sdn_b_enc
        self.sdn_b_dec = sdn_b_dec
        self.sdn_n_layers = sdn_n_layers
        self.feat_start = feat_start

        self.loss = loss
        self.viz = viz
        self.n_gpus = n_gpus
        self.exp_name = exp_name


    def forward(self, x):
        pass

    def training_step(self, batch, batch_nb):
        
        # Log loss
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat,y)
        self.log('train_loss', loss, prog_bar=False, logger=True)

        if self.global_step % self.viz == 0: #Visualization
            self._images_logger(x, y, y_hat,'train')
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):

        # Log loss
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.loss(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=False, logger=True)

        # Log images
        if self.global_step % self.viz == 0:
            self._images_logger(x, y, y_hat,'val')

        # Log evaluation metrics
        y_hat = (torch.sigmoid(y_hat) > 0.5).int()
        y = y.int()
        eval_measure = eval_metrics(y_hat, y, self.n_classes)
        self.log('val_dice', eval_measure['avg_dice'], prog_bar=False, logger=True)
        self.log('val_jaccard', eval_measure['avg_jaccard'], prog_bar=False, logger=True)

        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, logger=True,prog_bar=False)

    def test_step(self,batch, batch_nb):
        x, y= batch
        y_hat = self.forward(x)
        y_hat = (torch.sigmoid(y_hat) > 0.5).int()
        y = y.int()

        eval_measure = eval_metrics(y_hat,y,self.n_classes)
        print(y_hat.shape,'dice', eval_measure['avg_dice'])
        print(y_hat.shape, 'jaccard', eval_measure['avg_jaccard'])
        self._write_csv_out(avg_dice=eval_measure['avg_dice'],
                            avg_jaccard=eval_measure['avg_jaccard'],
                            avg_hd_95 = eval_measure['avg_hd_95'],
                            avg_hd_100 =eval_measure['avg_hd_100'],
                            batch_size=y_hat.shape[0])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)

    def _images_logger(self,x,y,y_hat,tag):
        batch_size = x.shape[0]
        n_samples = min(batch_size, 2)

        y_hat = torch.sigmoid(y_hat)

        for i in range(n_samples):
            if tag == 'train':
                nb = 0
            else:
                nb = 1

            self.logger.experiment.add_image(str(nb)+'/'+tag+'_images',x[i],self.global_step)
            self.logger.experiment.add_image(str(nb)+'/'+tag+'_masks',y[i], self.global_step)
            self.logger.experiment.add_image(str(nb)+'/'+tag+'_pred_masks', y_hat[i],self.global_step)

    def _write_csv_out(self, avg_dice, avg_jaccard, avg_hd_95, avg_hd_100, batch_size):
        time.sleep(3)

        if not os.path.exists('outputs'):
            os.mkdir('outputs')

        if not os.path.exists(os.path.join('outputs', self.exp_name[:-10])):
            os.makedirs(os.path.join('outputs', self.exp_name[:-10]), exist_ok=True)

        with open(os.path.join('outputs', self.exp_name +'.txt'), mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([avg_dice.cpu().numpy(),
                                 avg_jaccard.cpu().numpy(),
                                 avg_hd_95,
                                 avg_hd_100,
                                 batch_size,
                                 self.n_gpus])




    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--n_channels', type=int, default=3)
        parser.add_argument('--n_classes', type=int, default=1)
        parser.add_argument('--n_layers', type=int, default=5)
        parser.add_argument('--feat_start', type=int, default=32)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--bilinear', type=str2bool, default=True)
        parser.add_argument('--block', type=str, default='vgg_block')
        parser.add_argument('--sdn_b_enc', type=str2bool, default=False)
        parser.add_argument('--sdn_b_dec', type=str2bool, default=False)
        parser.add_argument('--sdn_n_layers', type=int, default=3)
        parser.add_argument('--loss', type=str, default='Dice_Loss')
        return parser

