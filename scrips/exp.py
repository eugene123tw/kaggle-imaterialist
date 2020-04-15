from mmdetection.mmdet.apis import init_detector, inference_detector, show_result

checkpoint_file = '/home/eugene/_MODELS/imaterialist/sota_imat_epoch_15.pth'
config_file = '/home/eugene/git/kaggle-imaterialist/configs/htc_dconv_c3-c5_mstrain_x101_64x4d_fpn_20e_1200x1900.py'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file)

# test a single image and show the results
img = '/home/eugene/_DATASETS/fashion/fashionista/test/natural_259_1.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
show_result(img, result, model.CLASSES, out_file="./result.jpg")

# test a list of images and write the results to image files
# imgs = ['/home/eugene/_DATASETS/fashion/fashionista/test/06_1_front.jpg', ]
# for i, result in enumerate(inference_detector(model, imgs)):
#     show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))