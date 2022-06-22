#!/usr/bin/env python3

import argparse
import numpy as np
import tensorflow as tf

from utils import uniform_sampler
from data_loader import data_loader, map_loader

parser = argparse.ArgumentParser(description='Test SNP GAIN')
parser.add_argument('--test_data', required=True, type=str)
parser.add_argument('--load_model', required=True, type=str)
parser.add_argument('--result', required=True, type=str)

args = parser.parse_args()

# Load data to npy
ori_data_x, miss_data_x, data_m = data_loader(args.test_data)
# data_map = map_loader('/content/drive/MyDrive/GAIN/data/TWB_compare/chr22_select_all.map')

data_m = 1 - np.isnan(ori_data_x)
norm_data_x = np.nan_to_num(ori_data_x, 0)

# Other parameters
no, dim = norm_data_x.shape

# Load model
# discriminator = tf.saved_model.load("./model/%s_D"%(args.load_model))
# generator = tf.saved_model.load("%s_G" % (args.load_model))
generator = tf.keras.models.load_model("%s_G" % (args.load_model))

# Return imputed data
Z_mb = uniform_sampler(0, 0.01, no, dim)
M_mb = tf.convert_to_tensor(data_m, dtype=tf.float32)
X_mb = tf.convert_to_tensor(norm_data_x, dtype=tf.float32)
X_mb = tf.cast(X_mb, dtype=tf.float32)
M_mb = tf.cast(M_mb, dtype=tf.float32)
X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

# Map_mb = np.repeat(data_map[np.newaxis, :], X_mb.shape[0], axis=0)
# Map_mb = tf.cast(Map_mb, dtype=tf.float32)

# imputed_data = generator([X_mb, M_mb, Map_mb])
imputed_data = generator([X_mb, M_mb])

imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data.numpy()

# Rounding
# imputed_data = imputed_data*2
# imputed_data = np.round_(imputed_data)
imputed_data = imputed_data.astype(int)

# Save result to hap
imputed_data = imputed_data.transpose()
np.savetxt(args.result, imputed_data, fmt='%i', delimiter=' ')
