"""
    模型准确度验证
    
"""
import tensorflow as tf
import os 
import numpy as np
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_path = '/home/fxf/resnet152/saved_Rest152_v2_model'
TestTFPath = '/home/fxf/TestDataset_generate/czrk/czrk_pair_test.tfrecord'
# TestTFPath = '/home/fxf/TestDataset_generate/glass_mask/glass_mask_pairs.tfrecord'
batch_size = 32
input_shape = 112

IMAGE_FEATURE_MAP = {
    'image1': tf.io.FixedLenFeature([], tf.string),
    'image2': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

def parse_tfrecord(single_record, input_shape):
    x = tf.io.parse_single_example(single_record, IMAGE_FEATURE_MAP)

    image1 = tf.image.decode_jpeg(x['image1'], channels=3)
    image1 = tf.image.resize(image1, (input_shape, input_shape))
    image1 = image1/255
    image2 = tf.image.decode_jpeg(x['image2'], channels=3)
    image2 = tf.image.resize(image2, (input_shape, input_shape))
    image2 = image2/255
    label = tf.stack([tf.cast(x['label'], 'int32')])
    return image1, image2, label

def load_tfrecord_dataset(file_pattern, input_shape):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, input_shape))


train_dataset = load_tfrecord_dataset(TestTFPath, input_shape)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=-1)

model = tf.keras.models.load_model(model_path)
print("load model successfully!")

distances = []
labels = []
for image1, image2, label in tqdm.tqdm(train_dataset):
    feature1 = model(image1)
    feature2 = model(image2)
    # feature1 = tf.math.l2_normalize(feature1, 1)
    # feature2 = tf.math.l2_normalize(feature2, 1)
    distance = tf.math.reduce_sum((feature1-feature2)**2, axis=1)
    for x in distance.numpy():
        distances.append(x)
    for x in label.numpy():
        labels.append(x[0])

distances = np.array(distances)
labels = np.array(labels)


threshold_nums = 1000
dists_len = len(distances)

th_base = np.min(distances)
th_step = (np.max(distances)-th_base)/threshold_nums
threshold_list = np.array([th_base+th_step*i for i in range(threshold_nums)])
acc_list = []
print("[*]阈值搜索中……")
for th in tqdm.tqdm(threshold_list):
    acc_count = 0
    for i, x in enumerate(distances):
        acc_count += int((x <= th) == labels[i])
    acc = acc_count/dists_len
    acc_list.append(acc)
acc_list = np.array(acc_list)
max_index = np.argmax(acc_list)
max_accu = acc_list[max_index]*100
th = th_base+max_index*th_step

print('模型识别准确率为: %.2f%%'%(max_accu))
print('临界阈值为: %.2f'%(th))
