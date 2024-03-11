# -*- coding: utf-8 -*-
import argparse
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.swa_utils import AveragedModel

from model.wide_res_net import WRN28_10
from model.pyramid import Pyramid
from model.resnet import ResNet18,ResNet50,ResNet101,ResNet152

from data.cifar import CIFAR10,CIFAR100


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--model',default='WRN28_10',type=str)
parser.add_argument('--dataset',default='CIFAR100',type=str)
#parser.add_argument('--optimizer',default='SGD',type=str,help='SGD')
parser.add_argument('--lr_type', default='CosineAnnealingLR', type=str)#'MultiStepLR'/'CosineAnnealingLR'
parser.add_argument('--epoch', default=400, type=int)
parser.add_argument('--start_averaged', default=300, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--weight_decay', default=5e-3, type=float)
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--eta_min',default=0.00,type=float)
parser.add_argument('--num_worker', default=8, type=int, help='number of workers')
parser.add_argument('--label_smoothing',default=0.0,type=float)
parser.add_argument('--loss_func', default='CE', type=str)
#'SCE/CE'



args = parser.parse_args()
print('model:',args.model)
print('dataset:',args.dataset)
#print('optimizer:',args.optimizer)
print('lr_type:',args.lr_type)
print('epoch:', args.epoch)
print('start_averaged:',args.start_averaged)
print('batch_size:', args.batch_size)
print('weight_decay:',args.weight_decay)
print('momentum:', args.momentum)
print('learning_rate:', args.lr)
print('eta_min:',args.eta_min)
print('num_worker:',args.num_worker)
print('label_smoothing:',args.label_smoothing)
print('loss_func:',args.loss_func)


dataset = eval(args.dataset)(args.batch_size, args.num_worker)
num_class = 10 if args.dataset == 'CIFAR10' else 100
print('num_class:',num_class)

print('==> Building model..')
model = eval(args.model)(num_class)
model = torch.nn.DataParallel(model)
averaged_model = AveragedModel(model)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.empty_cache()
    model.cuda()
    averaged_model.cuda()
    cudnn.benchmark = True
print('==> Finish model')


optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.lr_type == "MultiStepLR":
    scheduler = eval(args.lr_type)(optimizer, milestones=[args.epoch*2/10, args.epoch*4/10, args.epoch*6/10], gamma=args.eta_min)
else:
    scheduler = eval(args.lr_type)(optimizer = optimizer, T_max = args.epoch, eta_min = args.eta_min)

if args.loss_func == "SCE":
    loss_func = lambda output, target : smooth_crossentropy(output, target, smoothing=args.label_smoothing)
else:
    loss_func = torch.nn.CrossEntropyLoss()

total_train = 0
for inputs,_ in dataset.train:
    total_train += inputs.size()[0]
print('train_size:',total_train)

total_test = 0
for inputs,_ in dataset.test:
    total_test += inputs.size()[0]
print('test_size:',total_test)

def train(epoch):
    
    torch.cuda.synchronize()
    time_ep = time.time()
    model.train()
    averaged_model.train()

    train_loss = 0.0
    correct = 0.0    

    for inputs, target in dataset.train:
        optimizer.zero_grad()
        input_size = inputs.size()[0]
        inputs, target = inputs.cuda(), target.cuda()
        output = model(inputs)
        loss = loss_func(output, target)
        
        if args.loss_func == "SCE":
            loss.mean().backward()
            optimizer.step()
            if epoch >= args.start_averaged:
                averaged_model.update_parameters(model)
            with torch.no_grad():
                train_loss += loss.sum().item()/total_train
                correct += (torch.argmax(output, 1) == target).sum().item()/total_train
        else:
            loss.backward()
            optimizer.step()
            if epoch >= args.start_averaged:
                averaged_model.update_parameters(model)
            with torch.no_grad():
                train_loss += loss.data * input_size/total_train
                _, pred = torch.max(output, 1)
                correct += pred.eq(target).sum()/total_train            
    torch.cuda.synchronize()   
    time_ep = time.time() - time_ep
    
    if epoch >= args.start_averaged:
        torch.optim.swa_utils.update_bn(dataset.train, averaged_model, optimizer.param_groups[0]['params'][0].device)
        
            
    return train_loss, 100*correct, time_ep

def test(epoch):
    model.eval()
    averaged_model.eval()

    test_loss = 0.0
    test_correct = 0.0
    avg_test_loss = 0.0
    avg_test_correct = 0.0

    with torch.no_grad():
        for inputs, target in dataset.test:
            input_size = inputs.size()[0]
            if use_cuda:
                inputs, target = inputs.cuda(), target.cuda()
                
            if args.loss_func == "SCE":
                #normal_prediction
                predict = model(inputs)
                loss = loss_func(predict, target)
                test_loss += loss_func(predict, target).sum().item()/ total_test
                test_correct += (torch.argmax(predict, 1) == target).sum().item() / total_test
                #averaged_prediction
                avg_predict = averaged_model(inputs)
                loss = loss_func(avg_predict, target)
                avg_test_loss += loss_func(avg_predict, target).sum().item()/ total_test
                avg_test_correct += (torch.argmax(avg_predict, 1) == target).sum().item() / total_test
            else:              
                #normal_prediction
                predict = model(inputs)
                loss = loss_func(predict, target)
                test_loss += loss.data * input_size / total_test
                _, pred = torch.max(predict.data, 1)
                test_correct += pred.eq(target).sum() / total_test
                #averaged_prediction
                avg_predict = averaged_model(inputs)
                loss = loss_func(avg_predict, target)
                avg_test_loss += loss.data * input_size / total_test
                _, pred = torch.max(avg_predict.data, 1)
                avg_test_correct += pred.eq(target).sum() / total_test

    return test_loss, 100*test_correct, avg_test_loss, 100*avg_test_correct
  
total_time = 0
for epoch in range(args.epoch):
    train_loss, train_acc, time_ep = train(epoch)
    total_time += time_ep
    scheduler.step()
    print("epoch", epoch+1, "lr:{:.7f}".format(optimizer.param_groups[0]['lr']), " train_loss:{:.5f}".format(train_loss), "train_acc:{:.2f}".format(train_acc), "time:{:.3f}".format(time_ep))
    if (epoch + 1)%5 == 0:
        test_loss, test_acc, avg_test_loss, avg_test_acc= test(epoch)
        print('test_loss:{:.5f}'.format(test_loss), 'test_acc:{:.2f}'.format(test_acc), 'avg_test_loss:{:.5f}'.format(avg_test_loss), 'avg_test_acc:{:.2f}'.format(avg_test_acc))
print('{:.0f}:{:.0f}:{:.0f}'.format(total_time//3600, total_time%3600//60, total_time%3600%60))