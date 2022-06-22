#!/usr/bin/env python3

import argparse
import numpy as np
import tensorflow as tf

from utils import uniform_sampler, binary_sampler, normalization
from data_loader import data_loader, map_loader
from models import Discriminator, Generator

parser = argparse.ArgumentParser(description='Train SNP GAIN')
parser.add_argument('--ref_pannel', required=True, type=str,
                    help='Reference pannel with various formats.')
parser.add_argument('--save_model', required=True, type=str,
                    help='Filename for saved model.')
parser.add_argument('--batch_size',
                    help='the number of samples in mini-batch',
                    default=128,
                    type=int)
parser.add_argument('--hint_rate',
                    help='hint probability',
                    default=0.9,
                    type=float)
parser.add_argument('--miss_rate',
                    help='missing probability',
                    default=0.9,
                    type=float)
parser.add_argument('--alpha',
                    help='hyperparameter',
                    default=100,
                    type=float)
parser.add_argument('--epochs',
                    help='number of epoch',
                    default=5,
                    type=int)
args = parser.parse_args()

# Load data to npy
ori_data_x, miss_data_x, data_m = data_loader(args.ref_pannel, args.miss_rate)
# data_map = map_loader('/content/drive/MyDrive/GAIN/data/TWB_compare/chr22_select_all.map')

# Define mask matrix (0 denotes missing)
data_m = 1 - np.isnan(miss_data_x)
miss_data_x = np.nan_to_num(miss_data_x, 0)

# Other parameters
no, dim = miss_data_x.shape

# Data preprocessing
norm_data_x_batch = tf.data.Dataset.from_tensor_slices((miss_data_x, data_m))
norm_data_x_batch = norm_data_x_batch.shuffle(buffer_size=no)
norm_data_x_batch = norm_data_x_batch.batch(args.batch_size)

# Training loop
discriminator = Discriminator(int(dim))
generator = Generator(int(dim))
D_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
G_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
D_loss_list = []
G_loss_list = []
G_grad = []
G_lr = []
for epoch in range(args.epochs):
    for step, (X_mb, M_mb) in enumerate(norm_data_x_batch):
        # Map_mb = np.repeat(data_map[np.newaxis, :], X_mb.shape[0], axis=0)
        # Map_mb = tf.cast(Map_mb, dtype=tf.float32)

        X_mb = tf.cast(X_mb, dtype=tf.float32)
        M_mb = tf.cast(M_mb, dtype=tf.float32)
        Z_mb = uniform_sampler(0, 0.01, tf.shape(M_mb)[0], dim)

        X_mb = X_mb + (1 - M_mb) * Z_mb
        
        # G_sample = generator([X_mb, M_mb, Map_mb], training=False)
        G_sample = generator([X_mb, M_mb], training=False)
        Hat_X = X_mb + G_sample * (1 - X_mb)

        H_mb_temp = binary_sampler(args.hint_rate, tf.shape(M_mb)[0], dim)
        H_mb = M_mb * tf.convert_to_tensor(H_mb_temp, dtype=np.float32)

        with tf.GradientTape() as tape:
            D_prob = discriminator([Hat_X, H_mb], training=True)
            D_loss = -tf.reduce_mean(M_mb * tf.math.log(D_prob + 1e-8) + (1 - M_mb)
                                     * tf.math.log(1. - D_prob + 1e-8))
        D_loss_list.append(D_loss)
        grads = tape.gradient(D_loss, discriminator.trainable_weights)
        D_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

        with tf.GradientTape() as tape:
            # G_sample = generator([X_mb, M_mb, Map_mb], training=True)
            G_sample = generator([X_mb, M_mb], training=True)

            G_loss_temp = -tf.reduce_mean((1 - M_mb) * tf.math.log(D_prob + 1e-8))
            MSE_loss = tf.reduce_mean((M_mb * X_mb - M_mb * G_sample)**2) / tf.reduce_mean(M_mb)
            G_loss = G_loss_temp + args.alpha * MSE_loss
        G_loss_list.append(G_loss)
        grads = tape.gradient(G_loss, generator.trainable_weights)
        G_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
        G_lr.append(G_optimizer._decayed_lr('float32').numpy())
        curr_grad = []
        for i in grads:
            curr_grad.append(np.linalg.norm(np.array(i)))
        G_grad.append(curr_grad)

        if step % 10 == 0:
            print('Epoch:{}\tStep:{}\tD_loss:{:.2f}\tG_loss:{:.2f}'.format(epoch, step, D_loss, G_loss))

# Save model
discriminator.save("./model/%s_D" % (args.save_model))
generator.save("./model/%s_G" % (args.save_model))
# tf.saved_model.save(discriminator, "./model/%s_D" % (args.save_model))
# tf.saved_model.save(generator, "./model/%s_G" % (args.save_model))

# Save loss
D_loss_list = np.array(D_loss_list)
G_loss_list = np.array(G_loss_list)
G_lr = np.array(G_lr)
G_grad = np.array(G_grad)
np.save("./model/%s_D_loss.npy" % (args.save_model), D_loss_list)
np.save("./model/%s_G_loss.npy" % (args.save_model), G_loss_list)
np.save("./model/%s_G_lr.npy" % (args.save_model), G_lr)
np.save("./model/%s_G_grad.npy" % (args.save_model), G_grad)
