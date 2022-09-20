import tensorflow as tf
import os 
import numpy as np
import tqdm
from itertools import combinations
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

testdataset_dir="/home/fxf/sdk-update/bin/query_mask_aligned"
txt_save_path = '/home/fxf/TestDataset_generate/glass_mask/glass_mask_pairs.txt'

class_list = next(os.walk(testdataset_dir))[1]
class_list.sort()
class_num=len(class_list)

pair_list=[]
n_num=0
p_num=0
NP_proportion=0.3
for face_class in class_list:
    class_dir=os.path.join(testdataset_dir,face_class)
    face_list=os.listdir(class_dir)
    comb = list(combinations(face_list, 2))
    pair_num=len(comb)
    p_num+=pair_num
    for i in comb:
        pair_list.append((os.path.join(class_dir,i[0]),os.path.join(class_dir,i[1]),1))

n_num=int(p_num*NP_proportion)
print(n_num)
for i in range(n_num):
    member1 = class_list[np.random.randint(0, class_num)]
    member2 = class_list[np.random.randint(0, class_num)]
    if member1 == member2:
        continue
    img1_dir=os.path.join(testdataset_dir, member1)
    img2_dir=os.path.join(testdataset_dir, member2)
    img_1=os.listdir(img1_dir)[0]
    img_2=os.listdir(img2_dir)[0]
    # print(os.path.join(img2_dir,img_2))
    pair_list.append((os.path.join(img1_dir,img_1), os.path.join(img2_dir,img_2), 0))

    # save pairs(list) to a txt 
    with open(txt_save_path, 'w') as f:
        for pair in pair_list:
            f.write("{} {} {}\n".format(pair[0], pair[1], pair[2]))

"""
    Part Two:
    生成tfrecord格式的测试文件
"""
TestTFPath = '/home/fxf/TestDataset_generate/glass_mask/glass_mask_pairs.tfrecord'

def build_example(image1, image2, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image1])),
        'image2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image2])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))
    return example

# read pairs(list) from the txt 
with open(txt_save_path, 'r') as f:
    data = f.readlines()
    
writer = tf.io.TFRecordWriter(TestTFPath)
for pair in tqdm.tqdm(data):
    img1_path, img2_path, label = pair.split(' ')
    label = int(label[0])
    image1 = open(img1_path, 'rb').read()
    image2 = open(img2_path, 'rb').read()
    tf_example = build_example(image1, image2, label)
    writer.write(tf_example.SerializeToString())
writer.close()

print("测试集tfrecord文件创建完毕!")