#!/usr/bin/env python3

""" Evulate imputation accuracy.
    Example Usage:
        ./accuracy.py \
        --test ../data/GRCh37_impute2/test_0503/chr22_test_sub_gain_3500.hap \
        --mask ../data/GRCh37_impute2/test_0503/chr22_test_sub_test.hap.gz \
        --true ../data/GRCh37_impute2/test_0503/chr22_test_sub.hap.gz \
        --maf 1 0.1 0.01 0.001 0.0001 0 \
        --output ../result/0509_chr22_test_sub_accuracy_gain_3500.csv

        ./accuracy.py \
        --test ../data/TWB_compare/test_0503/chr22_test_TWB_gain_3500.hap \
        --mask ../data/TWB_compare/test/chr22_test_TWB_test.hap.gz \
        --true ../data/TWB_compare/test/chr22_test_TWB.hap.gz \
        --maf 1 0.1 0.01 0.001 0.0001 0 \
        --output ../result/0503_chr22_test_TWB_accuracy_gain_3500.csv
@version:1.0
@date:2022/02/14
"""

import os
import gzip
import argparse
import numpy as np
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="Mean of empty slice.")

parser = argparse.ArgumentParser(description='Evulate imputation accuracy.')
parser.add_argument('--test', required=True, type=str, help='Imputed result')
parser.add_argument('--mask', required=True, type=str, help='Missing genpotype')
parser.add_argument('--true', required=True, type=str, help='True genpotype')
parser.add_argument('--maf', required=True, type=float, nargs='+', help='MAF criteria')
parser.add_argument('--output', required=False, type=str, help='output file for accuracy')
args = parser.parse_args()


def smart_open(filename, mode="r"):
    if filename.endswith(".gz"):
        return(gzip.open(filename, mode))
    else:
        return(open(filename, mode))


def read_data(inFile):
    filename, file_extension = os.path.splitext(inFile)

    if file_extension == '.gz':
        filename, file_extension = os.path.splitext(filename)

    if file_extension == '.hap':
        data = smart_open(inFile)
        data = np.genfromtxt(data, delimiter=' ')
    elif file_extension == '.gen':
        raise ValueError('.gen file cannot compute aaccuracy.')
        ### .gen file cannot compute accuracy ###
        # data = []
        # with smart_open(inFile) as f:
        #     line = f.read()
        #     line = line.split(" ")
        #     line = line[5:]
        #     data.append(line)
        # data = np.array(data)
    else:
        raise ValueError('Not supported data format.')

    ### DATA IS TRANSPOSED!!! ###
    # row represent a smaple
    # column represent SNPs
    data = data.transpose()

    return data


# ===
# Read files
# ===
data_test = read_data(args.test)
data_mask = read_data(args.mask)
data_true = read_data(args.true)
no, dim = data_test.shape

print(data_test.shape)
print(data_mask.shape)
print(data_true.shape)

# ===
# Compute MAF
# ===
maf = np.mean(data_true, axis=0)
maf_bin = sorted(args.maf)
inds = np.digitize(maf, maf_bin, right=True)

# Reprot MAF bin
print("\n### MAF Report ###")
print('\nAll SNPs')
for i in range(max(inds)+1):
    print("<= %s : %s" % (maf_bin[i], np.sum(inds == i)))

print('\nTesting SNPs')
for i in range(max(inds)+1):
    test = data_test[0][(np.isnan(data_mask[0])) & (inds == i)]
    print("<= %s : %s" % (maf_bin[i], len(test)))

# ===
# Calculate accuracy
# ===
if args.output != None:
    all_accuracy = []

    # Header line
    # all_accuracy.append(maf_bin)

    for i in range(no):
        accuracy = []

        # Acc. for MAF bin
        for bin in range(len(maf_bin)):
            test = data_test[i][(np.isnan(data_mask[i])) & (inds == bin)]
            true = data_true[i][(np.isnan(data_mask[i])) & (inds == bin)]
            accuracy.append(accuracy_score(test, true))
        # Overall acc. for individual.
        test = data_test[i][np.isnan(data_mask[i])]
        true = data_true[i][np.isnan(data_mask[i])]
        accuracy.append(accuracy_score(test, true))

        # Append to final record
        all_accuracy.append(accuracy)
    all_accuracy = np.array(all_accuracy)
    np.savetxt(args.output, all_accuracy, delimiter=',')
