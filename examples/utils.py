import albumentations as albu
import PIL.Image as Image
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

import warnings
warnings.filterwarnings('ignore')

def get_data():
    # データを読み込む
    image = np.array(Image.open('../img/Lenna.png'))
    image_seg = np.array(Image.open('../img/0004a4c0-d4dff0ad.jpg'))
    label_seg = np.array(Image.open('../img/0004a4c0-d4dff0ad_train_id.png'))
    image_det = np.array(Image.open('../img/0000f77c-6257be58.jpg'))
    with open('../img/0000f77c-6257be58.json') as f:
        label_det = json.load(f)

    # segmentation maskのクラスをカラーで分ける
    mask = np.zeros((720, 1280, 3)).astype('uint8')
    color = np.array([51,51,51])
    for i, v in enumerate(np.unique(label_seg)):
        if i % 3 == 0:
            color[0] += 51
        elif i % 3 == 1:
            color[1] += 51
        elif i % 3 == 2:
            color[2] += 51
        mask[label_seg == v] = color

    # BDD100kのフォーマットからpascal vocのフォーマットに変換
    bboxes = get_bboxes(label_det['labels'])['bbox']
    
    data = {'image' : image,
            'seg_img' : image_seg,
            'mask' : mask,
            'det_img': image_det,
            'bboxes' : bboxes}
    return data


def get_bboxes(label):
    # BDD100kのフォーマットからpascal vocのフォーマットに変換
    bboxes = []
    names = []
    ids = []
    for lab in label:
        if 'box2d' in lab.keys():
            x_min = lab['box2d']['x1']
            x_max = lab['box2d']['x2']
            y_min = lab['box2d']['y1']
            y_max = lab['box2d']['y2']
            bboxes.append([x_min, y_min, x_max, y_max]) # set pascal_voc format
            names.append(lab['category'])
            ids.append(lab['id'])
    return {'bbox':bboxes, 'name': names, 'id': ids}


def imshow2(img, transforms=None):
    if transforms is None:
        print('Transforms is None')
        tsf = img
    else:
        tsf_dict = transforms(image=img)
        tsf = tsf_dict['image']
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
    ax[0].tick_params(labelbottom="off",bottom="off")
    ax[0].tick_params(labelleft="off",left="off")
    ax[1].tick_params(labelbottom="off",bottom="off")
    ax[1].tick_params(labelleft="off",left="off")
    ax[0].set_title('Original [{} x {}]'.format(img.shape[0], img.shape[1]))
    ax[1].set_title('Transformed [{} x {}]'.format(tsf.shape[0], tsf.shape[1]))
    ax[0].imshow(img.astype('uint8'))
    ax[1].imshow(tsf.astype('uint8'))
    fig.show()

    
def imshow4(img, mask, transforms=None):
    if transforms is None:
        print('Transforms is None')
        tsf = img
        tsf_m = mask
    else:
        tsf_dict = transforms(image=img, mask=mask)
        tsf, tsf_m = tsf_dict['image'], tsf_dict['mask']
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(14,7))
    ax[0,0].tick_params(labelbottom="off",bottom="off")
    ax[0,0].tick_params(labelleft="off",left="off")
    ax[1,0].tick_params(labelbottom="off",bottom="off")
    ax[1,0].tick_params(labelleft="off",left="off")
    ax[0,1].tick_params(labelbottom="off",bottom="off")
    ax[0,1].tick_params(labelleft="off",left="off")
    ax[1,1].tick_params(labelbottom="off",bottom="off")
    ax[1,1].tick_params(labelleft="off",left="off")
    ax[0, 0].set_title('Original image [{} x {}]'.format(img.shape[0], img.shape[1]))
    ax[1, 0].set_title('Transformed image [{} x {}]'.format(tsf.shape[0], tsf.shape[1]))
    ax[0, 1].set_title('Original mask [{} x {}]'.format(mask.shape[0], mask.shape[1]))
    ax[1, 1].set_title('Transformed mask [{} x {}]'.format(tsf_m.shape[0], tsf_m.shape[1]))

    ax[0, 0].imshow(img.astype('uint8'))
    ax[1, 0].imshow(tsf.astype('uint8'))
    ax[0, 1].imshow(mask)
    ax[1, 1].imshow(tsf_m)
    fig.show()
    
    
def putBbox(image, bboxes, thickness=3):
    img = image.copy()
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = list(map(int,bbox))
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=thickness)
    return img


