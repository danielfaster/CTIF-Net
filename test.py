# encoding=UTF-8
import torch
import torch.nn.functional as F
import numpy as np
import os
import torchvision.utils as vutils
from torch.autograd import Variable
from evaluate.data import test_dataset
import time
import torchvision.transforms as transforms
from model.CTIFNet import  CTIFNet
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=224, help='test size')
parser.add_argument('--trainset', type=str, default='DUTS-TR', help='training  dataset')
parser.add_argument('--mae_model_path', type=str, default='./models/checkpoint/mae_pretrain_vit_large.pth', help='mae pretrain model path')
parser.add_argument('--pre_model_path', type=str, default='./models/CTIFNet_w_.pth', help='pre-trained model path')
parser.add_argument('--dataset_path', type=str, default='./data/TestDatasets', help='test dataset path')
parser.add_argument('--test_datasets', nargs='+', type=str, default=['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S'], help='List of test datasets')

args = parser.parse_args()
print("====================config file is >>>>>>>> "+str(args))

print("Test datasets {} begin!".format(args.test_datasets))

model = CTIFNet(args)
model.cuda()

model.load_state_dict(torch.load(args.pre_model_path))
print("====== load model from {} success! ======".format(args.pre_model_path))
result_file_name="saliency_preds"


for i, dataset in enumerate(args.test_datasets):

    startTime = time.time()
    print("test NO:{} datasets {} begin test!".format(i,dataset))
    path_prefix = './'+result_file_name+'/'
    pre_image_merge_save_path = path_prefix + dataset + '/'
    os.makedirs(pre_image_merge_save_path,exist_ok=True)
    image_root = os.path.join(args.dataset_path, dataset)

    if dataset == "ECSSD" or dataset =="PASCAL-S":
        image_path = image_root + "/Imgs/"
        gt_path = image_path
    elif dataset == "DUTS-TE":
        image_path = image_root + "/DUTS-TE-Image/"
        gt_path = image_root + "/DUTS-TE-Mask/"
    elif dataset == "DUTS-TR":
        image_path = image_root + "/DUTS-TR-Image/"
        gt_path = image_root + "/DUTS-TR-Mask/"
    elif dataset == "DUTS-TR":
        image_path = image_root + "/DUTS-TR-Image/"
        gt_path = image_root + "/DUTS-TR-Mask/"
    else:
        image_path = image_root + "/Img/"
        gt_path = image_root + "/GT/"

    test_loader = test_dataset(image_path, gt_path, args.testsize, dataset)
    total_test_time =0
    total_time =0
    model.eval()
    for i in range(test_loader.size):
        # if i>10 and dataset == "DUTS-TE":
        #     break
        image_tran, gt, orignin_image, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = Variable(image_tran).cuda()
        begin_time = time.time()
        result_res, result_trans = model(image)
        test_end_time = time.time()
        total_test_time +=(test_end_time-begin_time)

        result_res = F.interpolate(result_res, size=gt.shape, mode='bilinear', align_corners=False)
        result_trans = F.interpolate(result_trans, size=gt.shape, mode='bilinear', align_corners=False)
        temp_name = name.split('.png')[0]
        transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        ]
        )

        vutils.save_image(result_res, pre_image_merge_save_path + temp_name + ".png")


        save_end_time = time.time()
        total_time += (save_end_time-begin_time)


    print(" ave_test_time is {:.4f} s  total_test_time is {:.4f} s".format(total_test_time/test_loader.size,total_test_time))
    print("total_time is {} s ".format(total_time))
    print("Test  dataset {} [Cost:{:.4f}s] finished! ".format(dataset,(time.time()-startTime)))



