import os
from argparse import ArgumentParser
import csv

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from lib.models import UNet, NestedUnet
from lib.core import ToyNet
from lib.dataset import PL_DSB2018, PL_Polyp, PL_LiTS, PL_segTHOR
from lib.utils import str2bool


def main(hparams):

    os.makedirs(hparams.log_dir, exist_ok=True)

    log_dir = os.path.join(hparams.log_dir, hparams.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(log_dir, 'checkpoints'),
        monitor='val_loss',
        save_top_k=1,
        mode = 'min',
        period=3
    )

    if hparams.early_stopping is True:
        stop_callback = EarlyStopping(
            monitor='avg_val_loss',
            min_delta=0.0,
            patience=30,
            mode='min',
            verbose=False,
            strict = True,
        )
    else:
        stop_callback = []

    # Model
    models = dict({'unet': UNet,
                   'nested_unet': NestedUnet})
    model_name = models[hparams.model]
    model = model_name(**vars(hparams))

    #print(model)

    # Dataset
    if hparams.dataset == 'dsb2018':
        datamodule = PL_DSB2018(hparams.datadir,hparams.batch_size, hparams.kfold)

    elif hparams.dataset == 'polyp':
        datamodule = PL_Polyp(hparams.datadir,hparams.batch_size, hparams.kfold)

    elif hparams.dataset == 'lits':
        datamodule = PL_LiTS(hparams.datadir,hparams.batch_size, hparams.kfold)

    elif hparams.dataset == 'segthor':
        datamodule = PL_segTHOR(hparams.datadir,hparams.batch_size, hparams.kfold)

    else:
        raise Exception('dataset not implemented or not recognized')

    trainer = Trainer(
            val_check_interval=4,
            precision=16,
            amp_level='01',
            gpus=hparams.n_gpus,
            distributed_backend='ddp',
            checkpoint_callback=checkpoint_callback,
            callbacks=[stop_callback],
            logger=TensorBoardLogger(save_dir=hparams.log_dir, name=hparams.exp_name),
            progress_bar_refresh_rate=20,
            max_epochs=hparams.max_epochs
        )

    if hparams.train:
        trainer.fit(model, datamodule)

    if hparams.test:
        trainer.test(model, test_dataloaders=datamodule.test_loader())


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--datadir', default='data')
    parent_parser.add_argument('--log_dir', default='logs')
    parent_parser.add_argument('--exp_name', default='baseline')
    parent_parser.add_argument('--train', type=str2bool, default=True)
    parent_parser.add_argument('--test', type=str2bool, default=False)
    parent_parser.add_argument('--max_epochs', type=int, default=300)
    parent_parser.add_argument('--checkpoint', default=str)
    parent_parser.add_argument('--checkpoint_hparams', default=str)
    parent_parser.add_argument('--n_gpus',type=int, default=4)
    parent_parser.add_argument('--batch_size', type=int)
    parent_parser.add_argument('--model', type=str)
    parent_parser.add_argument('--viz', type=int, default=20)
    parent_parser.add_argument('--early_stopping', type=str2bool, default=True)
    parent_parser.add_argument('--kfold', type=int, default=-1)


    parser = ToyNet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
