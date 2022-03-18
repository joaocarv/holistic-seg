import os
import cv2
import numpy as np
import SimpleITK as sitk
import pandas as pd
from torch.utils.data import Dataset, Subset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split


class DSB2018_Dataset(Dataset):
    def __init__(self, ds_dir, set=None, kfold=None, transform=None):

        self.transform = transform
        self.img_dir = os.path.join(ds_dir, 'images')
        self.msk_dir = os.path.join(ds_dir, 'masks/0')
        if kfold == None:
            self.img_list = os.listdir(self.img_dir)
        else:
            # ds_dir = r'dataset/dsb2018_96'
            csv_file = ds_dir + '/cv/'+str(kfold) +'/'+ set + '.csv'
            self.img_list = pd.read_csv(csv_file).iloc[:,0].tolist()


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        img_id = self.img_list[i]
        img_file = os.path.join(self.img_dir, img_id )
        msk_file = os.path.join(self.msk_dir, img_id)
        img = cv2.imread(img_file)
        msk = cv2.imread(msk_file, cv2.IMREAD_GRAYSCALE)[..., None]

        if self.transform is not None:
            augmented = self.transform(image=img, mask=msk)
            img = augmented['image']
            msk = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        msk = msk.astype('float32') / 255
        msk = msk.transpose(2, 0, 1)

        return img, msk, img_id


class Polyp_Dataset(Dataset):
    def __init__(self, ds_dir, set=None, kfold=None, transform=None, subset=None):
        # ds_dir = 'dataset//polyp//CVC-ClinicDB/'
        # ds_dir = 'dataset//polyp//ETIS-LaribPolypDB'

        self.transform = transform
        self.img_dir = os.path.join(ds_dir, 'Original')
        self.msk_dir = os.path.join(ds_dir, 'Ground Truth')
        self.img_list = os.listdir(self.img_dir)

        self.etis = ('ETIS' in ds_dir)

        if kfold:
            ds_dir = r'dataset/polyp'
            csv_file = ds_dir +'/cv/'+ str(kfold) + '/' +set + '.csv'
            self.img_list = pd.read_csv(csv_file).iloc[:,0].tolist()

        elif subset and not self.etis:
            metadata_df = pd.read_csv(os.path.join(ds_dir,'metadata.csv'))

            if subset == 'train':
                train_ids = metadata_df.sequence_id.unique()[6:]
                train_ids = metadata_df[metadata_df.sequence_id.isin(train_ids)].frame_id.tolist()
                train_ids = [str(x)+'.tif' for x in train_ids]

                self.img_list = [x for x in self.img_list if x in train_ids]

            if subset == 'val':
                val_ids = metadata_df.sequence_id.unique()[3:6]
                val_ids = metadata_df[metadata_df.sequence_id.isin(val_ids)].frame_id.tolist()
                val_ids = [str(x)+'.tif' for x in val_ids]
                self.img_list = [x for x in self.img_list if x in val_ids]

            if subset == 'test':
                test_ids = metadata_df.sequence_id.unique()[3:6]
                test_ids = metadata_df[metadata_df.sequence_id.isin(test_ids)].frame_id.tolist()
                test_ids = [str(x)+'.tif' for x in test_ids]
                self.img_list = [x for x in self.img_list if x in test_ids]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        img_id = self.img_list[i]
        if self.etis:
            msk_id = 'p' + img_id
        else:
            msk_id = img_id
        img_file = os.path.join(self.img_dir, img_id)
        msk_file = os.path.join(self.msk_dir, msk_id)
        img = cv2.imread(img_file)
        msk = cv2.imread(msk_file, cv2.IMREAD_GRAYSCALE)[..., None]
        img = cv2.resize(img, (128, 96))
        msk = cv2.resize(msk, (128, 96))[..., None]

        if self.transform is not None:
            augmented = self.transform(image=img, mask=msk)
            img = augmented['image']
            msk = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        msk = msk.astype('float32') / 255
        msk = msk.transpose(2, 0, 1)

        return img, msk, img_id


class LiTS_Dataset(Dataset):
    def __init__(self, ds_dir, set=None, kfold=None, transform=None, subset=None):

        self.transform = transform
        self.img_dir = os.path.join(ds_dir, 'images')
        self.msk_dir = os.path.join(ds_dir, 'masks')
        self.img_list = os.listdir(self.img_dir)
        if kfold == None:

            if subset:
                ids = np.linspace(28,130, 130)
                n_test = int(ids.size*0.1)

                if subset == 'train':
                    train_ids =ids[n_test:].tolist()
                    self.img_list = [x for x in self.img_list if (int(x.split('_')[0]) in train_ids)]

                if subset == 'val':
                    val_ids = ids[:n_test].tolist()
                    self.img_list = [x for x in self.img_list if (int(x.split('_')[0]) in val_ids)]

                if subset == 'test':
                    test_ids = ids[0:n_test].tolist()
                    self.img_list = [x for x in self.img_list if (int(x.split('_')[0]) in test_ids)]
        else:
            # ds_dir = r'dataset/LiTS_128'
            csv_file = ds_dir + '/cv/' + str(kfold) + '/' + set + '.csv'
            self.img_list = pd.read_csv(csv_file).iloc[:, 0].tolist()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        img_id = self.img_list[i]
        img_file = os.path.join(self.img_dir, img_id )
        msk_file = os.path.join(self.msk_dir, img_id)
        img = sitk.ReadImage(img_file)
        msk = sitk.ReadImage(msk_file)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=msk)
            img = augmented['image']
            msk = augmented['mask']

        img_np = sitk.GetArrayFromImage(img).astype('float32')
        img_np_norm_clip = np.squeeze(img_np)
        img_np_norm_clip[img_np_norm_clip < -1000] = -1000
        img_np_norm_clip[img_np_norm_clip > 1000] = 1000
        img_np_norm_clip = (img_np_norm_clip + 1000)/2000.
        # img_3chn_np = img_np_norm_clip

        # img_3chn_np = np.array([img_np_norm_clip, img_np_norm_clip, img_np_norm_clip])
        img_3chn_np = img_np_norm_clip[None, ...]
        msk_np = sitk.GetArrayFromImage(msk).astype('float32')
        msk_np[msk_np == 2] = 1

        return img_3chn_np, msk_np, img_id


class segTHOR_Dataset(Dataset):
    def __init__(self, ds_dir, set=None, kfold=None, transform=None):

        self.transform = transform
        self.img_dir = os.path.join(ds_dir, 'images')
        self.msk_dir = os.path.join(ds_dir, 'masks')

        if kfold == None:
            self.img_list = os.listdir(self.img_dir)
        else:
            csv_file = ds_dir + '/cv/'+str(kfold) +'/'+ set + '.csv'
            self.img_list = pd.read_csv(csv_file).iloc[:,0].tolist()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        img_id = self.img_list[i]
        img_file = os.path.join(self.img_dir, img_id )
        msk_file = os.path.join(self.msk_dir, img_id)
        img = sitk.ReadImage(img_file)
        msk = sitk.ReadImage(msk_file)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=msk)
            img = augmented['image']
            msk = augmented['mask']

        img_np = sitk.GetArrayFromImage(img).astype('float32')
        img_np_norm_clip = np.squeeze(img_np)
        img_np_norm_clip[img_np_norm_clip < -1000] = -1000
        img_np_norm_clip[img_np_norm_clip > 1000] = 1000
        img_np_norm_clip = (img_np_norm_clip + 1000)/2000.
        img_3chn_np = np.array([img_np_norm_clip, img_np_norm_clip, img_np_norm_clip])
        msk_np = sitk.GetArrayFromImage(msk).astype('float32')
        msk_np[msk_np == 2] = 1

        msk_np = np.moveaxis(msk_np,-1,0)
        # if msk_np.shape != [1,128,256]:
        return img_3chn_np, msk_np, img_id


class DataSubSet(Dataset):
    '''
    Dataset wrapper to apply transforms separately to subsets
    '''
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        img, msk, img_id = self.subset[i]
        if self.transform:
            augmented = self.transform(image=img, mask=msk)
            img = augmented['image']
            msk = augmented['mask']
        return img, msk, img_id



class PL_DSB2018(LightningDataModule):
    def __init__(self, data_dir, batch_size, kfold):
        super().__init__()

        self.dir_images = os.path.join(data_dir,'dsb2018')
        self.train_dir = os.path.join(self.dir_images,'train')
        self.batch_size = batch_size
        self.kfold = kfold

        self.train_transform=None
        self.val_transform=None

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        # called on every GPU
        if self.kfold == -1:

            dataset = DSB2018_Dataset(ds_dir=self.train_dir)

            # Split dataset
            n_test = int(len(dataset) * 0.1)
            n_val = n_test
            n_train = int(len(dataset) - n_test*2)
            idx = list(range(len(dataset)))

            test_ds =  Subset(dataset, idx[:n_test])
            dataset_nontest = Subset(dataset, idx[n_test:])
            train_ds, val_ds = random_split(dataset_nontest, [n_train, n_val])


        else:
            non_testset = DSB2018_Dataset(ds_dir=self.train_dir, set='train', kfold=self.kfold)
            n_val = int(len(non_testset) * 0.1)
            n_train = int(len(non_testset) - n_val)

            train_ds, val_ds = random_split(non_testset, [n_train, n_val])
            test_ds = DSB2018_Dataset(ds_dir=self.train_dir, set='test', kfold=self.kfold)

        self.train_ds = DataSubSet(train_ds, transform=self.train_transform)
        self.val_ds =  DataSubSet(val_ds, transform=self.val_transform)
        self.test_ds = DataSubSet(test_ds, transform=self.val_transform)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size= self.batch_size,
            pin_memory = True,
            shuffle = True,
            num_workers = 4
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=5
        )
        return val_loader

    def test_loader(self):
        test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=5
        )
        return test_loader


class PL_Polyp(LightningDataModule):
    def __init__(self, data_dir, batch_size, kfold):
        super().__init__()
        self.dir_images = os.path.join(data_dir, 'polyps')
        self.train_dir = os.path.join(self.dir_images, 'CVC-ClinicDB')
        self.test_dir = os.path.join(self.dir_images, 'ETIS-LaribPolypDB')
        self.batch_size = batch_size
        self.kfold = kfold

        self.train_transform = None
        self.val_transform = None
        self.use_external = False

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # called on every GPU

        # Split dataset
        if self.kfold == -1:
            if self.use_external:
                dataset = Polyp_Dataset(ds_dir=self.train_dir)
                n_val = int(len(dataset) * 0.1)
                n_train = len(dataset) - n_val

                train_ds, val_ds = random_split(dataset, [n_train, n_val])

                self.train_ds = DataSubSet(train_ds, transform=self.train_transform)
                self.val_ds = DataSubSet(val_ds, transform=self.val_transform)
                self.test_ds = Polyp_Dataset(ds_dir=self.test_dir, transform = self.val_transform)
            else:
                # Split dataset patient-wise
                self.train_ds = Polyp_Dataset(ds_dir=self.train_dir, transform=self.train_transform, subset='train')
                self.val_ds = Polyp_Dataset(ds_dir=self.train_dir, transform=self.val_transform, subset='val')
                self.test_ds = Polyp_Dataset(ds_dir=self.train_dir, transform=self.val_transform, subset='test')

        else:
            train_ds = Polyp_Dataset(ds_dir=self.train_dir, set='train', kfold=self.kfold)
            val_ds = Polyp_Dataset(ds_dir=self.train_dir, set='validation', kfold=self.kfold)
            test_ds = Polyp_Dataset(ds_dir=self.train_dir, set='test', kfold=self.kfold)

            self.train_ds = DataSubSet(train_ds, transform=self.train_transform)
            self.val_ds = DataSubSet(val_ds, transform=self.val_transform)
            self.test_ds = DataSubSet(test_ds, transform=self.val_transform)


    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=4
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=4
        )
        return val_loader

    def test_loader(self):
        test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=4
        )
        return test_loader


class PL_LiTS(LightningDataModule):
    def __init__(self, data_dir, batch_size, kfold=None):
        super().__init__()
        self.dir_images = os.path.join(data_dir,'lits')
        self.batch_size = batch_size
        self.kfold = kfold
        self.train_transform = None
        self.val_transform = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # called on every GPU

        # Split dataset
        if self.kfold == -1:
            # self.train_ds = LiTS_Dataset(ds_dir=self.train_dir, transform=self.train_transform, subset='train')
            # self.val_ds = LiTS_Dataset(ds_dir=self.train_dir, transform=self.val_transform, subset='val')
            # self.test_ds = LiTS_Dataset(ds_dir=self.test_dir, transform=self.val_transform)
            self.train_ds = LiTS_Dataset(ds_dir=self.dir_images, transform=self.train_transform, subset='train')
            self.val_ds = LiTS_Dataset(ds_dir=self.dir_images, transform=self.val_transform, subset='val')
            self.test_ds = LiTS_Dataset(ds_dir=self.dir_images, transform=self.val_transform)


        else:
            self.train_ds = LiTS_Dataset(ds_dir=self.dir_images, set='train', kfold=self.kfold, transform=self.train_transform)
            self.val_ds = LiTS_Dataset(ds_dir=self.dir_images, set='validation', kfold=self.kfold, transform=self.train_transform)
            self.test_ds = LiTS_Dataset(ds_dir=self.dir_images, set='test', kfold=self.kfold, transform=self.train_transform)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=2
        )

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds,
                                batch_size=self.batch_size,
                                pin_memory=True,
                                shuffle=False,
                                num_workers=2)
        return val_loader

    # def test_dataloader(self):
    def test_loader(self):
        test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=4
        )
        return test_loader



class PL_segTHOR(LightningDataModule):
    def __init__(self, data_dir, batch_size, kfold):
        super().__init__()
        self.dir_images = os.path.join(data_dir, 'segTHOR')

        self.train_dir = os.path.join(self.dir_images, 'train')
        self.val_dir = os.path.join(self.dir_images, 'val')
        self.test_dir = os.path.join(self.dir_images, 'test')
        self.batch_size = batch_size
        self.kfold = kfold

        self.train_transform = None
        self.val_transform = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # called on every GPU

        if self.kfold == -1:
            dataset_train = segTHOR_Dataset(ds_dir=self.train_dir)
            dataset_val = segTHOR_Dataset(ds_dir=self.val_dir)
            dataset_test = segTHOR_Dataset(ds_dir=self.test_dir)


            train_ds = dataset_train
            val_ds = dataset_val
            test_ds = dataset_test

            self.train_ds = DataSubSet(train_ds, transform=self.train_transform)
            self.val_ds = DataSubSet(val_ds, transform=self.val_transform)
            self.test_ds = DataSubSet(test_ds, transform=self.val_transform)

        else:
            train_ds = segTHOR_Dataset(ds_dir=self.dir_images, set='train', kfold=self.kfold)
            val_ds = segTHOR_Dataset(ds_dir=self.dir_images, set='validation', kfold=self.kfold)
            test_ds = segTHOR_Dataset(ds_dir=self.dir_images, set='test', kfold=self.kfold)

            self.train_ds = DataSubSet(train_ds, transform=self.train_transform)
            self.val_ds = DataSubSet(val_ds, transform=self.val_transform)
            self.test_ds = DataSubSet(test_ds, transform=self.val_transform)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=5
        )

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds,
                                batch_size=self.batch_size,
                                pin_memory=True,
                                shuffle=False,
                                num_workers=5)
        return val_loader

    def test_loader(self):
        test_loader = DataLoader(self.test_ds,
                                 batch_size=self.batch_size,
                                 pin_memory=True,
                                 shuffle=False,
                                 num_workers=2)
        return test_loader