# Copyright (c) OpenMMLab. All rights reserved.
from imageio import imread

import numpy as np
import rawpy

from mmgen.datasets.builder import PIPELINES



@PIPELINES.register_module()
class LoadRAWFromFile:
    """Load image from file.

    Args:
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 key='gt'):
        self.key = key

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        filepath = str(results[f'{self.key}_path'])
        if filepath.endswith('TIF'):
            img = imread(filepath).astype(np.float32)
        else:
            with rawpy.imread(filepath) as f:
                img = f.raw_image_visible.copy().astype(np.float32)

        results[self.key] = img
        results[f'{self.key}_path'] = filepath
        results[f'{self.key}_ori_shape'] = img.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(key={self.key})'
        return repr_str


@PIPELINES.register_module()
class Demosaic:
    """ Demosaic RAW image.
    """

    def __init__(self,
                 key='gt'):
        self.key = key

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = results[self.key]
        if len(img.shape) == 3:
            assert img.shape[2] == 1
            img = img[..., 0]
        h, w = img.shape
        rgb = np.zeros((h, w, 3), dtype=img.dtype)

        img = np.pad(img, ((2, 2), (2, 2)), mode='reflect')
        r, gb, gr, b = img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]
        
        rgb[0::2, 0::2, 0] = r[1:-1, 1:-1]
        rgb[0::2, 0::2, 1] = (gr[1:-1, 1:-1] + gr[:-2, 1:-1] + gb[1:-1, 1:-1] + gb[1:-1, :-2]) / 4
        rgb[0::2, 0::2, 2] = (b[1:-1, 1:-1] + b[:-2, :-2] + b[1:-1, :-2] + b[:-2, 1:-1]) / 4

        rgb[1::2, 0::2, 0] = (r[1:-1, 1:-1] + r[2:, 1:-1]) / 2
        rgb[1::2, 0::2, 1] = gr[1:-1, 1:-1]
        rgb[1::2, 0::2, 2] = (b[1:-1, 1:-1] + b[1:-1, :-2]) / 2

        rgb[0::2, 1::2, 0] = (r[1:-1, 1:-1] + r[1:-1, 2:]) / 2
        rgb[0::2, 1::2, 1] = gb[1:-1, 1:-1]
        rgb[0::2, 1::2, 2] = (b[1:-1, 1:-1] + b[:-2, 1:-1]) / 2

        rgb[1::2, 1::2, 0] = (r[1:-1, 1:-1] + r[2:, 2:] + r[1:-1, 2:] + r[2:, 1:-1]) / 4
        rgb[1::2, 1::2, 1] = (gr[1:-1, 1:-1] + gr[1:-1, 2:] + gb[1:-1, 1:-1] + gb[2:, 1:-1]) / 4
        rgb[1::2, 1::2, 2] = b[1:-1, 1:-1]

        results[self.key] = rgb
        results[f'{self.key}_ori_shape'] = rgb.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(key={self.key})'
        return repr_str


@PIPELINES.register_module()
class RAWNormalize:
    """Rearrange RAW to four channels.
    """

    def __init__(self,
                 key='gt',
                 blc=528,
                 saturate=4095):
        self.key = key
        self.blc = blc
        self.saturate = saturate

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = results[self.key]
        img = (img-self.blc) / (self.saturate-self.blc)
        img = np.clip(img, 0, 1)
        results[self.key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(key={self.key}, ' \
                    f'(blc={self.blc}, ' \
                    f'(saturate={self.saturate})'
        return repr_str
