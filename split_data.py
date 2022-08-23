import data_loader

path = 'train/chr22_train_TWB_3500.hap'
data,miss_data,data_m = data_loader.data_loader(path)
print(len(data),len(miss_data))
print(data_m)
