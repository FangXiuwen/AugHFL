from __future__ import print_function

import argparse
import os
import shutil
import time

from Dataset import augmentations
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  mixture_width = 3
  mixture_depth = -1
  aug_severity = 3
  all_ops = True

  aug_list = augmentations.augmentations
  if all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(mixture_width):
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


class AugMixDataset(Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return aug(x, self.preprocess), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)


class AugMixPublicDataset(Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, corrupt='fix'):
    self.dataset = dataset
    self.preprocess = preprocess
    self.corrupt = corrupt

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.corrupt == 'fix':
      return aug(x, self.preprocess), y
    elif self.corrupt == 'random':
      im_tuple = (aug(x, self.preprocess), aug(x, self.preprocess),
                  aug(x, self.preprocess), aug(x, self.preprocess))
      return im_tuple, y
    elif self.corrupt == 'augmix':
      im_tuple1 = (self.preprocess(x), aug(x, self.preprocess),aug(x, self.preprocess))
      im_tuple2 = (self.preprocess(x), aug(x, self.preprocess),aug(x, self.preprocess))
      im_tuple3 = (self.preprocess(x), aug(x, self.preprocess),aug(x, self.preprocess))
      im_tuple4 = (self.preprocess(x), aug(x, self.preprocess),aug(x, self.preprocess))
      im_tuple = (im_tuple1, im_tuple2, im_tuple3, im_tuple4)
      return im_tuple, y
    elif self.corrupt == 'fixaugmix':
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),aug(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)