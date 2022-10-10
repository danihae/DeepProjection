import deepprojection as dp
import os
import shutil
import numpy as np
import tifffile

# create test folder with random training and test data

folder = './temp_test/'
folder_image = folder + 'training_data/image/'
folder_mask = folder + 'training_data/mask/'
folder_data = folder + 'data/'
folder_results = folder + 'results/'
os.makedirs(folder, exist_ok=True)
os.makedirs(folder_mask, exist_ok=True)
os.makedirs(folder_image, exist_ok=True)
os.makedirs(folder_data, exist_ok=True)
os.makedirs(folder_results, exist_ok=True)

for i in range(5):
    # regular unet
    random_image = np.random.randint(0, 255, (8, 128, 128))
    random_mask = np.random.randint(0, 1, (8, 128, 128)) * 255
    tifffile.imwrite(folder_image + f'{i}.tif', random_image)
    tifffile.imwrite(folder_mask + f'{i}.tif', random_mask)


random_movie = np.random.randint(0, 255, (20, 8, 128, 128))
tifffile.imwrite(folder + 'movie.tif', random_movie)

# create training data set
data = dp.DataProcess(source_dirs=(folder_image, folder_mask), n_slices=10, dim_out=(64, 64), data_path=folder+'data/')

# train
train = dp.Trainer(data, num_epochs=4, n_filter=8, n_slices=10, save_dir=folder + 'models/')
train.start()


# predict movie
predict = dp.Project(folder + 'movie.tif', data_format='TZXY', weights=folder + 'models/model_best.pth', mode='mip',
                     filename_output=folder_results + 'movie.tif', resize_dim=(64, 64))


# delete test folder
shutil.rmtree(folder)


