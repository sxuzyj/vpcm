import ast
import io
import logging
from typing import Optional
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
import os
from timm.data.readers import create_reader

_logger = logging.getLogger(__name__)
from torch.utils.data import Dataset
import os

_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            input_img_mode='RGB',
            transform=None,
            target_transform=None,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.input_img_mode = input_img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.reader[index]

        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.input_img_mode and not self.load_bytes:
            img = img.convert(self.input_img_mode)
        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class MultiModalImageDataset(Dataset):
    def __init__(self , image_dir,csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            text_column (string): Name of the column containing text data.
            label_column (string): Name of the column containing labels.
            transform (callable, optional): Optional transform to be applied
                on an image sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        # self.text_column = text_column
        # self.label_column = label_column
        self.transform = transform
        # super().__init__(root=image_dir, transform=transform)

        self.data_dir = image_dir
        self.transform = transform
        self.data = []

        for index, row in self.data_frame.iterrows():
            year_month = row['image_name'].split(']')[1].split('.')[0].split('_')

            # 将年份和月份转换为整数
            year = int(year_month[0])
            month = int(year_month[1])

            # 创建列表
            date = [year, month]
            image_path = os.path.join(self.image_dir, str(row['label']))
            image_path = os.path.join(image_path, row['image_name'])
            label = row['label']
            Temperature = torch.tensor(ast.literal_eval(row['Temperature']))
            location = ast.literal_eval(row['Location'])
            # Temperature_MAX = torch.tensor(ast.literal_eval(row['Temperature_MAX']))
            # Temperature_MIN = torch.tensor(ast.literal_eval(row['Temperature_MIN']))
            Precipitation_SUM = torch.tensor(ast.literal_eval(row['Precipitation_SUM']))
            # Precipitation_MAX = torch.tensor(ast.literal_eval(row['Precipitation_MAX']))
            # Precipitation_MIN = torch.tensor(ast.literal_eval(row['Precipitation_MIN']))
            # self.data.append((image_path, Temperature, Temperature_MAX, Temperature_MIN, Precipitation_SUM, Precipitation_MAX, Precipitation_MIN, label))
            self.data.append((image_path, Temperature,  Precipitation_SUM, label, location, date))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # image_path, t, t_max, t_min, p, p_max, p_min , label = self.data[idx]
        image_path, t,  p,  label , Location , date= self.data[idx]

        # 加载并转换图像
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 文本数据可能需要额外的处理，这里假设我们直接返回原始文本
        # return image, t, t_max, t_min, p, p_max, p_min , label
        return image, t, p,  label, Location, date


#
# class MultiModalImageDataset(ImageDataset):
#     def __init__(self , image_dir,csv_file, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             image_dir (string): Directory with all the images.
#             text_column (string): Name of the column containing text data.
#             label_column (string): Name of the column containing labels.
#             transform (callable, optional): Optional transform to be applied
#                 on an image sample.
#         """
#         self.data_frame = pd.read_csv(csv_file)
#         self.image_dir = image_dir
#         # self.text_column = text_column
#         # self.label_column = label_column
#         self.transform = transform
#         super().__init__(root=image_dir, transform=transform)
#
#     def __len__(self):
#         return len(self.data_frame)
#
#     def __getitem__(self, idx):
#         try:
#             if torch.is_tensor(idx):
#                 idx = idx.tolist()
#
#             img_name, img, target = super().__getitem__(idx)
#             # img = self.loader(img_path)
#             # if self.transform is not None:
#             #     img = self.transform(img)
#
#             directory, filename = os.path.split(img_name)
#
#             row = self.data_frame.loc[self.data_frame['image_name'] == filename]
#
#             Temperature = ast.literal_eval(row['Temperature'].values[0])
#             Temperature_MAX = ast.literal_eval(row['Temperature_MAX'].values[0])
#             Temperature_MIN = ast.literal_eval(row['Temperature_MIN'].values[0])
#             Precipitation_SUM = ast.literal_eval(row['Precipitation_SUM'].values[0])
#             Precipitation_MAX = ast.literal_eval(row['Precipitation_MAX'].values[0])
#             Precipitation_MIN = ast.literal_eval(row['Precipitation_MIN'].values[0])
#
#             # sample = {
#             #     'image': img,
#             #     'Temperature': Temperature,
#             #     'Temperature_MAX': Temperature_MAX,
#             #     'Temperature_MIN': Temperature_MIN,
#             #     'Precipitation_SUM': Precipitation_SUM,
#             #     'Precipitation_MAX': Precipitation_MAX,
#             #     'Precipitation_MIN': Precipitation_MIN
#             # }
#             # return sample, target
#
#             return img, Temperature, Temperature_MAX, Temperature_MIN, Precipitation_SUM, Precipitation_MAX, Precipitation_MIN, target
#         except Exception as e:
#             print(f"Error processing index {idx}: {e}")
#             raise e
#

