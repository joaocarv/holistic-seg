import torch
from torch import nn
from lib.core import Up, Down, return_block, VGGBlock, VGGBlockResMixSDN, ToyNet
from lib import utils

class UNet(ToyNet):

    def __init__(self,  n_channels, n_classes, n_layers, lr, feat_start, bilinear, block, sdn_b_enc,sdn_b_dec, sdn_n_layers, loss, **kw):
        super().__init__(n_channels, n_classes, n_layers, lr, feat_start, bilinear, block, sdn_b_enc,sdn_b_dec, sdn_n_layers, loss, **kw)

        self.n_channels = n_channels
        self.n_layers = n_layers + 1
        feats = feat_start

        # Loss function
        self.loss = eval('utils.' + loss + '(self.n_channels, feats)')

        layers = [return_block(block)(self.n_channels, feats)]


        for i in range(self.n_layers - 1): # downsampling
            if self.sdn_b_enc is True and i <= sdn_n_layers:
                block_enc = block + '_sdn'
            else:
                block_enc = block
            layers.append(Down(feats, feats * 2, block_enc))
            feats *= 2

        for i in range(self.n_layers - 1): # upsampling
            if self.sdn_b_dec is True and i <= sdn_n_layers:
                block_dec = block + '_sdn'
            else:
                block_dec = block
            layers.append(Up(feats, feats // 2, block_dec, bilinear))
            feats //= 2


        layers.append(nn.Conv2d(feats, n_classes, kernel_size=1))
        self.layers = nn.ModuleList(layers)


    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1:self.n_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.n_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])



class NestedUnet(ToyNet):
    def __init__(self, n_channels, n_classes, n_layers, feat_start, bilinear, block, sdn_b_dec, sdn_b_enc, sdn_n_layers, loss, **kw):
        super().__init__(n_channels, n_classes, n_layers, feat_start, bilinear, block, sdn_b_dec, sdn_b_enc, sdn_n_layers, loss, **kw)

        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = False

        # Loss function
        self.loss = eval('utils.' + loss + '(self.n_channels, feat_start)')


        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        # Encoder
        if sdn_b_enc:
            self.conv0_0 = VGGBlockResMixSDN(n_channels, nb_filter[0])
            self.conv1_0 = VGGBlockResMixSDN(nb_filter[0], nb_filter[1])
            if self.n_layers >= 2:
                if  sdn_n_layers>=2:
                    self.conv2_0 = VGGBlockResMixSDN(nb_filter[1], nb_filter[2])
                else:
                    self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
            if self.n_layers >= 3:
                if  sdn_n_layers>=3:
                    self.conv3_0 = VGGBlockResMixSDN(nb_filter[2], nb_filter[3])
                else:
                    self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3])

            if self.n_layers >= 4:
                if sdn_n_layers>=4:
                    self.conv4_0 = VGGBlockResMixSDN(nb_filter[3], nb_filter[4])
                else:
                    self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4])
        else:
            self.conv0_0 = VGGBlock(n_channels, nb_filter[0])
            self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
            if self.n_layers >= 2:
                self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
            if self.n_layers >= 3:
                self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3])
            if self.n_layers >= 4:
                self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4])


        if sdn_b_dec:
            self.conv0_1 = VGGBlockResMixSDN(nb_filter[0] + nb_filter[1], nb_filter[0])
            if self.n_layers >= 2:
                if sdn_n_layers>=2:
                    self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
                    self.conv0_2 = VGGBlockResMixSDN(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
                else:
                    self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
                    self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
            if self.n_layers >= 3:
                if sdn_n_layers >= 3:
                    self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
                    self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
                    self.conv0_3 = VGGBlockResMixSDN(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
                else:
                    self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
                    self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
                    self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
            if self.n_layers >=4:
                if  sdn_n_layers>=4:
                    self.conv3_1 = VGGBlockResMixSDN(nb_filter[3] + nb_filter[4], nb_filter[3])
                    self.conv2_2 = VGGBlockResMixSDN(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])
                    self.conv1_3 = VGGBlockResMixSDN(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])
                    self.conv0_4 = VGGBlockResMixSDN(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
                else:
                    self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
                    self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])
                    self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])
                    self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        else:
            self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
            if self.n_layers >= 2:
                self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
                self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
            if self.n_layers >= 3:
                self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
                self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
                self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
            if self.n_layers >= 4:
                self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
                self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])
                self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])
                self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    def forward(self, input):

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        if self.n_layers >= 2:
            x2_0 = self.conv2_0(self.pool(x1_0))
            x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
            x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        if self.n_layers >= 3:
            x3_0 = self.conv3_0(self.pool(x2_0))
            x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
            x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
            x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        if self.n_layers >= 4:
            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            if self.n_layers == 2:
                output = self.final(x0_2)
            elif self.n_layers == 3:
                output = self.final(x0_3)
            elif self.n_layers == 4:
                output = self.final(x0_4)
            else:
                output = self.final(x0_1)
            return output



