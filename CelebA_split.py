import os
import shutil
from tqdm import tqdm
data_dir = '/data/CelebA/'
identifier_dir = data_dir + 'identity_CelebA.txt'

with open(identifier_dir, 'r') as f:
    identifier = f.read().splitlines()

# print(identifier[-1])
img_dir = data_dir + 'img_celeba/'
db_dir = data_dir + 'db_name/'
for id in tqdm(identifier):
    img_name, name = id.split()
    # print(img_name, name)

    if os.path.isdir(db_dir + name):
        shutil.move(img_dir + img_name, db_dir + name)
    else:
        os.makedirs(db_dir + name)
        shutil.move(img_dir + img_name, db_dir + name)
