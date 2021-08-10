import Augmentor
import os
from shutil import rmtree

input_dir = './data_set/mini_vww_img'
# output_dir = '../data_set/img_aug'
output_dir = ".../mini_vww_aug"

out_train_background_dir = os.path.join(output_dir, 'train', 'background')
out_train_person_dir = os.path.join(output_dir, 'train', 'person')
# out_val_background_dir = os.path.join(output_dir, 'val', 'background')
# out_val_person_dir = os.path.join(output_dir, 'val', 'person')

# 只增强train_set, 不增强 val_set
in_train_background_dir = os.path.join(input_dir, 'train', 'background')
in_train_person_dir = os.path.join(input_dir, 'train', 'person')

# in_val_background_dir = os.path.join(input_dir, 'val', 'background')
# in_val_person_dir = os.path.join(input_dir, 'val', 'person')


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def Augmentor_Handler(input_dir, output_dir, samples_num):
    p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
    p.rotate90(probability=0.5)
    p.rotate270(probability=0.5)
    p.flip_left_right(probability=0.8)
    p.flip_top_bottom(probability=0.3)
    p.crop_random(probability=0.5, percentage_area=0.5)
    p.sample(samples_num)
    print('done')
    
    
# mk_file(out_train_background_dir)
# mk_file(out_train_person_dir)
# mk_file(out_val_background_dir)
# mk_file(out_val_person_dir)

# 扩充5倍
Augmentor_Handler(in_train_background_dir, out_train_background_dir, 20000)
Augmentor_Handler(in_train_person_dir, out_train_person_dir, 20000)
# Augmentor_Handler(in_val_background_dir, 10000)
# Augmentor_Handler(in_val_person_dir, 10000)
