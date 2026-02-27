import argparse
import os
from shutil import copyfile


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def class_prefix_to_int(class_name):
    # CUB class folder format example: 001.Black_footed_Albatross
    try:
        return int(class_name.split('.')[0])
    except Exception:
        return None


parser = argparse.ArgumentParser(description='Prepare CUB-200-2011 dataset')
parser.add_argument('--path', default='./data/CUB', type=str, help='raw dataset root path')
opt = parser.parse_args()

download_path = opt.path
images_path = os.path.join(download_path, 'images')
if not os.path.isdir(images_path):
    raise FileNotFoundError(f'images folder not found: {images_path}')

train_all_path = os.path.join(images_path, 'train_all')
test_path = os.path.join(images_path, 'test')
ensure_dir(train_all_path)
ensure_dir(test_path)

# 1-100 train_all, 101-200 test
for name in os.listdir(images_path):
    src = os.path.join(images_path, name)
    if not os.path.isdir(src):
        continue
    if name in ('train_all', 'test', 'train', 'val'):
        continue
    class_id = class_prefix_to_int(name)
    if class_id is None:
        continue
    if class_id <= 100:
        dst = os.path.join(train_all_path, name)
    else:
        dst = os.path.join(test_path, name)
    if not os.path.isdir(dst):
        os.rename(src, dst)

# split train_all -> train/val (first image in each class goes to val)
train_path = os.path.join(images_path, 'train')
val_path = os.path.join(images_path, 'val')
ensure_dir(train_path)
ensure_dir(val_path)

for class_name in os.listdir(train_all_path):
    class_dir = os.path.join(train_all_path, class_name)
    if not os.path.isdir(class_dir):
        continue
    class_train_dir = os.path.join(train_path, class_name)
    class_val_dir = os.path.join(val_path, class_name)
    ensure_dir(class_train_dir)
    ensure_dir(class_val_dir)

    first = True
    for file_name in sorted(os.listdir(class_dir)):
        if not file_name.lower().endswith('.jpg'):
            continue
        src_path = os.path.join(class_dir, file_name)
        if first:
            dst_path = os.path.join(class_val_dir, file_name)
            first = False
        else:
            dst_path = os.path.join(class_train_dir, file_name)
        copyfile(src_path, dst_path)

print(f'Prepared CUB dataset at: {images_path}')

# export to standard pytorch structure expected by train.sh/train.py/test.py
save_path = os.path.join(download_path, 'pytorch')
train_save = os.path.join(save_path, 'train')
val_save = os.path.join(save_path, 'val')
train_all_save = os.path.join(save_path, 'train_all')
query_save = os.path.join(save_path, 'query')
gallery_save = os.path.join(save_path, 'gallery')
for p in [save_path, train_save, val_save, train_all_save, query_save, gallery_save]:
    ensure_dir(p)

for class_name in os.listdir(train_path):
    src_dir = os.path.join(train_path, class_name)
    if not os.path.isdir(src_dir):
        continue
    dst_dir = os.path.join(train_save, class_name)
    ensure_dir(dst_dir)
    for file_name in os.listdir(src_dir):
        if file_name.lower().endswith('.jpg'):
            copyfile(os.path.join(src_dir, file_name), os.path.join(dst_dir, file_name))

for class_name in os.listdir(val_path):
    src_dir = os.path.join(val_path, class_name)
    if not os.path.isdir(src_dir):
        continue
    dst_val = os.path.join(val_save, class_name)
    dst_all = os.path.join(train_all_save, class_name)
    ensure_dir(dst_val)
    ensure_dir(dst_all)
    for file_name in os.listdir(src_dir):
        if file_name.lower().endswith('.jpg'):
            src_file = os.path.join(src_dir, file_name)
            copyfile(src_file, os.path.join(dst_val, file_name))
            copyfile(src_file, os.path.join(dst_all, file_name))

for class_name in os.listdir(train_path):
    src_dir = os.path.join(train_path, class_name)
    if not os.path.isdir(src_dir):
        continue
    dst_all = os.path.join(train_all_save, class_name)
    ensure_dir(dst_all)
    for file_name in os.listdir(src_dir):
        if file_name.lower().endswith('.jpg'):
            src_file = os.path.join(src_dir, file_name)
            copyfile(src_file, os.path.join(dst_all, file_name))

for class_name in os.listdir(test_path):
    src_dir = os.path.join(test_path, class_name)
    if not os.path.isdir(src_dir):
        continue
    dst_q = os.path.join(query_save, class_name)
    dst_g = os.path.join(gallery_save, class_name)
    ensure_dir(dst_q)
    ensure_dir(dst_g)
    for file_name in os.listdir(src_dir):
        if file_name.lower().endswith('.jpg'):
            src_file = os.path.join(src_dir, file_name)
            copyfile(src_file, os.path.join(dst_q, file_name))
            copyfile(src_file, os.path.join(dst_g, file_name))

print(f'Prepared CUB pytorch folder at: {save_path}')
