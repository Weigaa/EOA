#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import numpy as np
import math
from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
torch.backends.cudnn.benchmark = True

data_root = '/data/imagenet/'
totalbsz = 512
numworker = 16
disturb = 0.01
num_epochs = 90
warmup = True

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def getbesthyperparameter(tensorlist,tensor):
    #share loss and tensor
    dist.all_gather(tensorlist,tensor)

def average_params(model):
    """ Params averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size

def broadcast_bestparams(model,src):
    """ Params broadcast. """
    for param in model.parameters():
        dist.broadcast(param.data, src)

#计算灰狼算法和鲸鱼算法用到的D、A
def getD(X_p,X_t):
    C = random.uniform(0,2)
    return abs(C*X_p-X_t)

def getA(t,t_max=num_epochs):
    #t_max设为epoch总数，默认为200
    r =  random.uniform(0,1)
    a = 2-2*(t/t_max)
    return a*(2*r-1)

#计算灰狼算法和鲸鱼算法用到的D'
def getD2(X_p,X_t):
    return abs(X_p-X_t)


#灰狼优化算法
def GWO(mytensorlist, bestlr, myrank, epoch, populationsize=4):
    GWOtensorlist=mytensorlist[0:populationsize]
    omega=[]
    for i in range(len(GWOtensorlist)):
        GWOtensorlist[i].append(i)
    #按准确率从大到小排序
    GWOtensorlist.sort(reverse=True)
    #从大到小分配种群节点等级
    alpha = GWOtensorlist[0]
    beta = GWOtensorlist[1]
    delta = GWOtensorlist[2]
    for i in range(3,populationsize):
        omega.append(GWOtensorlist[i])
    #更新alpha
    A_alpha_b = getA(t=epoch)
    D_alpha_b = getD(bestlr,alpha[1])
    #更新变化量dist
    mydist = A_alpha_b*D_alpha_b
    #dist进行归一化，以免偏移最优值过大
    mydist = ((mydist+4)/8) * disturb * bestlr
    alpha[1] = bestlr - mydist
    if myrank == alpha[2]:
        return abs(alpha[1])
    #更新beta
    A_beta_b = getA(t=epoch)
    D_beta_b = getD(bestlr, beta[1])
    A_beta_alpha = getA(t=epoch)
    D_beta_alpha = getD(alpha[1], beta[1])
    #更新变化量dist1、2
    dist1 = A_beta_b*D_beta_b
    dist2 = A_beta_alpha*D_beta_alpha
    #dist1、2进行归一化，以免偏移最优值过大
    dist1 = ((dist1 + 4)/8) * disturb * bestlr
    dist2 = ((dist2 + 4) / 8) * disturb * bestlr
    beta[1] = ((bestlr-dist1)+(alpha[1]-dist2))/2
    if myrank == beta[2]:
        return abs(beta[1])
    #更新delta
    A_delta_b = getA(t=epoch)
    D_delta_b = getD(bestlr, delta[1])
    A_delta_alpha = getA(t=epoch)
    D_delta_alpha = getD(alpha[1], delta[1])
    A_delta_beta = getA(t=epoch)
    D_delta_beta = getD(beta[1], delta[1])
    #更新变化量dist1、2、3
    dist1 = A_delta_b*D_delta_b
    dist2 = A_delta_alpha*D_delta_alpha
    dist3 = A_delta_beta*D_delta_beta
    #dist1、2进行归一化，以免偏移最优值过大
    dist1 = ((dist1 + 4) / 8) * disturb * bestlr
    dist2 = ((dist2 + 4) / 8) * disturb * bestlr
    dist3 = ((dist3 + 4) / 8) * disturb * bestlr
    delta[1] = ((bestlr-dist1)+(alpha[1]-dist2)+(beta[1]-dist3))/3
    if myrank == delta[2]:
        return abs(delta[1])
    #更新omega
    for i in range(len(omega)):
        if myrank == omega[i][2]:
            A_omegai_alpha = getA(t = epoch)
            D_omegai_alpha = getD(alpha[1], omega[i][1])
            A_omegai_beta = getA(t = epoch)
            D_omegai_beta = getD(beta[1], omega[i][1])
            A_omegai_delta = getA(t = epoch)
            D_omegai_delta = getD(delta[1], omega[i][1])
            # 更新变化量dist1、2、3
            dist1 = A_omegai_alpha * D_omegai_alpha
            dist2 = A_omegai_beta * D_omegai_beta
            dist3 = A_omegai_delta*D_omegai_delta
            # dist1、2进行归一化，以免偏移最优值过大
            dist1 = ((dist1 + 4) / 8) * disturb * bestlr
            dist2 = ((dist2 + 4) / 8) * disturb * bestlr
            dist3 = ((dist3 + 4) / 8) * disturb * bestlr
            omega[i][1] = ((alpha[1] - dist1) + (beta[1] - dist2)+(delta[1] - dist3)) / 3
            return omega[i][1]

#鲸鱼优化算法
def WOA(mylr, bestlr, epoch, mytensorlist, populationsize):
    WOAtensorlist =mytensorlist[dist.get_world_size()-populationsize:]
    rand = random.random()
    #更新公式中的b、l
    b = 1
    l = np.random.rand()
    if rand >= 0.5:
        D2 = getD2(bestlr, mylr)
        mylr = bestlr + D2*math.exp(b*l)*math.cos(2*math.pi*l)* disturb* bestlr
        return mylr
    else:
        A = getA(t = epoch)
        if abs(A) < 1:
            D = getD(bestlr,mylr)
            # 更新变化量dist
            mydist = A*D
            # dist进行归一化，以免偏移最优值过大
            mydist = ((mydist + 4) / 8) * disturb* bestlr
            mylr = bestlr - mydist
        else:
            randid = random.randint(0,populationsize-1)
            randlr = mytensorlist[randid][1]
            D = getD(randlr, mylr)
            # 更新变化量dist
            mydist = A*D
            # dist进行归一化，以免偏移最优值过大
            mydist = ((mydist + 4) / 8) * disturb* bestlr
            #randlr和bestlr差距过大时，调整randlr到合适的区间
            if randlr/bestlr > 2:
                randlr = randlr/int(randlr/bestlr)
            mylr = randlr-mydist
        return mylr

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_dataset():
    """ Partitioning ImageNet """
    traindir = os.path.join(data_root, 'train')
    assert os.path.exists(traindir), traindir + ' not found'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(
        traindir,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    size = dist.get_world_size()
    bsz = totalbsz / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    # train_set = torch.utils.data.DataLoader(
    #     partition, batch_size=int(bsz), shuffle=True, num_workers=numworker)
    train_set = DataLoaderX(
        partition, batch_size=int(bsz), shuffle=True, num_workers=numworker)
    return train_set, bsz


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        # dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=0)
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def run(rank, size):
    """ Distributed Synchronous SGD Example """
    device = torch.device("cuda:{}".format(rank % 2))
    basiclr = 0.1
    torch.manual_seed(1234)
    #确定数据集和模型
    train_set, bsz = partition_dataset()
    model = torchvision.models.resnet18(pretrained=False).to(device)
    model = model
    #初始化EOA参数
    # 灰狼种群参数
    GWOsize = 4
    # 鲸鱼种群参数
    WOAsize = dist.get_world_size() - GWOsize
    tensor_list = [torch.zeros(2, dtype=torch.float).to(device) for _ in range(dist.get_world_size())]
    mytensor_list = [[0,0] for _ in range(dist.get_world_size())]
    # 初始为不同节点增加0.9~1.1的扰动
    rand = 1.0 + 0.1 * np.random.rand()
    # GWO使用线性缩放，WOA使用根号缩放,total bsz为256时对应于0.1
    if rank < GWOsize:
        initbasiclr = basiclr * totalbsz/256
        mylr = initbasiclr * rand
    else:
        initbasiclr = basiclr * math.sqrt(totalbsz / 256)
        mylr = initbasiclr * rand
    #定义验证集
    valdir = os.path.join(data_root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    input_size = 224
    # test_loader = torch.utils.data.DataLoader(
    test_loader = DataLoaderX(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=int(bsz),
        shuffle=True,
        num_workers=numworker
    )
    #warm_up设置
    if warmup:
        mylrlist = [0.1, 0.1+1*(mylr-0.1)/4, 0.1+2*(mylr-0.1)/5, 0.1+3*(mylr-0.1)/4, mylr]
        mylr = mylrlist[0]
    optimizer = optim.SGD(model.parameters(), lr=mylr, momentum=0.9)
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    totalbegin = time.time()
    lastbestacc = 0
    bestacc = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        begin = time.time()
        # #debug变量k，提前结束循环
        # k = 0
        model.train()
        for data, target in train_set:
            # if k > 10:
            #     break
            # k += 1
            data, target = Variable(data), Variable(target)
#            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data).to(device)
            loss = F.cross_entropy(output, target).to(device)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        end = time.time()
        spendtime = (end-begin)/60
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches,', spend time: ',spendtime)
        if epoch >= 5:
            # 每个epoch结束后修改lr和momentum
            print("rank ",rank,"开始计算本节点模型的测试集准确率...")
            acc = get_accuracy(test_loader, model, device)
            # acc = rank
            print("rank ", rank, "节点测试集准确率为",acc)
            sharetensor = torch.tensor([acc,mylr]).to(device)
            getbesthyperparameter(tensor_list, sharetensor)
            for i in range(len(tensor_list)):
                mytensor_list[i] = tensor_list[i].tolist()
            bestrank = mytensor_list.index(max(mytensor_list))
            bestlr = mytensor_list[bestrank][1]
            bestacc = mytensor_list[bestrank][0]
            #手动实现余弦退火更新
            bestlr = (bestlr/(1+math.cos(epoch*math.pi/num_epochs)))*(1+math.cos((epoch+1)*math.pi/num_epochs))
            if rank == 0:
                print(mytensor_list)
                print("best rank is",bestrank,"best lr is",bestlr)
            # GWO计算下一epoch学习率
            if dist.get_rank() < GWOsize:
                mylr = abs(GWO(mytensorlist=mytensor_list, populationsize=GWOsize, bestlr=bestlr, myrank=dist.get_rank(), epoch=epoch))
            else:
                # WOA计算下一epoch学习率
                mylr = abs(WOA(mylr=mylr, bestlr = bestlr, epoch=epoch, mytensorlist=mytensor_list, populationsize=WOAsize))
            # 如果leader有用就用leader，否则用归约
            if bestacc > lastbestacc:
                broadcast_bestparams(model, bestrank)
            else:
                average_params(model)
            lastbestacc = bestacc
        if epoch < 5:
            mylr = mylrlist[epoch+1]
        print("rank ", rank, "epoch", epoch + 1, "learning rate is ", mylr)
        optimizer = optim.SGD(model.parameters(), lr=mylr, momentum=0.9)
    totalend = time.time()
    print("total realrun time: ", (totalend - totalbegin) / 60)
    get_accuracy(test_loader, model, device)

def get_accuracy(test_loader, model, device):
    model.eval()
    correct_sum = 0
    # Use GPU to evaluate if possible
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            out = model(data).to(device)
            pred = out.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum = correct_sum + correct
    acc = correct_sum / len(test_loader.dataset)
    # print("rank ",dist.get_rank()," acc is ",acc)
    return acc


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '28135'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 8
    processes = []
    print("execution beginning...")
    print("all process BEGINtime is : ", time.strftime('%Y-%m-%d %H:%M:%S'))
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print("all process ENDtime is : ", time.strftime('%Y-%m-%d %H:%M:%S'))
