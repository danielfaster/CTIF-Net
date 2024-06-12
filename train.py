#encoding=utf-8
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import os, argparse
from utils.func import AvgMeter
from evaluate.data import get_loader
from model.CTIFNet import  CTIFNet

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--step_size', type=int, default=20, help='step size')
parser.add_argument('--batchsize', type=int, default=4, help='batch size')
parser.add_argument('--trainsize', type=int, default=224, help='input size')
parser.add_argument('--trainset', type=str, default='DUTS-TR', help='training  dataset')
parser.add_argument('--mae_model_path', type=str, default='./models/checkpoint/mae_pretrain_vit_large.pth', help='mae pre-trained model path')

args = parser.parse_args()
print("====================config file is >>>>>>>> "+str(args))

#  backbone
dataset_path = './data/TrainDataset/'


image_root = dataset_path + args.trainset + '/DUTS-TR-Image/'
gt_root = dataset_path + args.trainset + '/DUTS-TR-Mask/'

model = CTIFNet(args)
model.cuda()

result_file_name = 'evaluate_test_results'
model_save_path = './models/'
os.makedirs(model_save_path,exist_ok=True)


# training
train_loader = get_loader(image_root, gt_root, batchsize=args.batchsize, trainsize=args.trainsize)
total_step = len(train_loader)
params = model.parameters()
optimizer = torch.optim.SGD(params, args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

CE = torch.nn.BCEWithLogitsLoss()


# training
for epoch in range(0, args.epoch):

    loss_record1, loss_record2 = AvgMeter(), AvgMeter()
    model.train()

    for i, pack in enumerate(train_loader, start=1):

        optimizer.zero_grad()
        images, gts, names = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        # forward
        y_res,y_tran = model(images)
        loss1 = CE(y_res, gts)
        loss2 = CE(y_tran, gts)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        loss_record1.update(loss1.data, args.batchsize)
        loss_record2.update(loss2.data, args.batchsize)

        if i % 200 == 0 or i == total_step:
            print(' Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_res: {:.4f}, Loss_trans: {:0.4f} '.
            format( epoch, args.epoch, i, total_step, loss_record1.show(), loss_record2.show()))


    scheduler.step()


save_model_path = model_save_path  + 'CTIFNet_w_.pth'
torch.save(model.state_dict(), save_model_path)
print(" save model success!")


print(" Congratulation! Training finished!!!")
