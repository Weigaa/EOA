import argparse
import torch

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar2 import Cifar2
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
import sys; sys.path.append("..")
from sam import SAM
import os
import torch.utils.data.distributed
import os
import torch
import torch.distributed as dist
import torch.utils.data.distributed
from torch.multiprocessing import Process
import numpy as np
import random
import math
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '18361'

def getbesthyperparameter(tensorlist,tensor):
    #share loss and tensor
    dist.all_gather(tensorlist,tensor)

def average_params(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        # dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=0)
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size

#计算灰狼算法和鲸鱼算法用到的D、A
def getD(X_p,X_t):
    C = random.uniform(0,2)
    return abs(C*X_p-X_t)

def getA(t,t_max=200):
    #t_max设为epoch总数，默认为200
    r =  random.uniform(0,1)
    a = 2-2*(t/t_max)
    return a*(2*r-1)

#计算灰狼算法和鲸鱼算法用到的D'
def getD2(X_p,X_t):
    return abs(X_p-X_t)


#灰狼优化算法
def GWO(mytensorlist, bestlr, myrank, epoch, populationsize=4, decay = 1):
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
    mydist = ((mydist+4)/8) * decay
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
    dist1 = ((dist1 + 4)/8) * decay
    dist2 = ((dist2 + 4) / 8) * decay
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
    dist1 = ((dist1 + 4) / 8) * decay
    dist2 = ((dist2 + 4) / 8) * decay
    dist3 = ((dist3 + 4) / 8) * decay
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
            dist1 = ((dist1 + 4) / 8) * decay
            dist2 = ((dist2 + 4) / 8) * decay
            dist3 = ((dist3 + 4) / 8) * decay
            omega[i][1] = ((alpha[1] - dist1) + (beta[1] - dist2)+(delta[1] - dist3)) / 3
            return omega[i][1]

#鲸鱼优化算法
def WOA(mylr, bestlr, epoch, mytensorlist, populationsize,decay = 1):
    WOAtensorlist =mytensorlist[dist.get_world_size()-populationsize:]
    rand = random.random()
    #更新公式中的b、l
    b = 1
    l = np.random.rand()
    if rand >= 0.5:
        D2 = getD2(bestlr, mylr)
        # 更新变化量dist
        mydist = D2*math.exp(b*l)*math.cos(2*math.pi*l)
        # dist进行归一化，以免偏移最优值过大
        mydist = ((mydist + 1.67) / 4.38) * decay
        mylr = bestlr + mydist
        return mylr
    else:
        A = getA(t = epoch)
        if abs(A) < 1:
            D = getD(bestlr,mylr)
            # 更新变化量dist
            mydist = A*D
            # dist进行归一化，以免偏移最优值过大
            mydist = ((mydist + 4) / 8) * decay
            mylr = bestlr - mydist
        else:
            randid = random.randint(0,populationsize-1)
            randlr = mytensorlist[randid][1]
            D = getD(randlr, mylr)
            # 更新变化量dist
            mydist = A*D
            # dist进行归一化，以免偏移最优值过大
            mydist = ((mydist + 4) / 8) * decay
            mylr = randlr-mydist
        return mylr



def main(rank):
    dist.init_process_group("gloo", rank=rank, world_size=args.train_workers)
    initialize(args, seed=42)
    torch.cuda.set_device(rank % 2)
    #灰狼种群参数
    GWOsize = 4
    #鲸鱼种群参数
    WOAsize = dist.get_world_size() - GWOsize
    # 针对性修改batchsize
    batchsize = int(args.batch_size / (args.train_workers / 2))
    print("each node batchsize is", batchsize)
    dataset = Cifar2(batchsize, args.threads)
    log = Log(log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank % 2])
    np.random.seed(rank)
    # mylr = args.learning_rate * (0.8 + 0.4 * np.random.rand())
    #为不同种群创建不同初始学习率
    #初始为不同节点增加0.9~1.1的扰动
    rand = 1.0 + 0.1 * np.random.rand()
    #GWO使用线性缩放，WOA使用根号缩放
    if rank < GWOsize:
        mylr = args.learning_rate * dist.get_world_size() * rand
    else:
        mylr = args.learning_rate * math.sqrt(dist.get_world_size()) * rand
    print("rank ",rank, "epoch 0","learning rate is ",mylr)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=mylr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    #引入余弦表优化器leader
    if rank != -1:
        optimizer2 = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=0.1,
                        momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, 200, eta_min=0, last_epoch=-1)

    #EOA算法用参数
    tensor_list = [torch.zeros(2, dtype=torch.float).cuda() for _ in range(dist.get_world_size())]
    mytensor_list = [[0, 0] for _ in range(dist.get_world_size())]

    for epoch in range(args.epochs):
        dataset.train_sampler.set_epoch(epoch)
        model.train()
        log.train(len_dataset=len(dataset.train))
        epoch_accuracy = 0
        for batch in dataset.train:
            inputs, targets = (b.cuda() for b in batch)

            # first forward-backward step
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            smooth_crossentropy(model(inputs), targets).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), mylr)
                # log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                # scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.cuda() for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
            epoch_accuracy = log.epoch_state["accuracy"] / log.epoch_state["steps"]
        average_params(model)
        # MPOS-1选举最优学习率（简化版）
        sharetensor = torch.tensor([epoch_accuracy, mylr]).cuda()
        getbesthyperparameter(tensor_list, sharetensor)
        for i in range(len(tensor_list)):
            mytensor_list[i] = tensor_list[i].tolist()
        if rank == 0:
            print(mytensor_list)
        bestrank = mytensor_list.index(max(mytensor_list))
        bestlr = mytensor_list[bestrank][1]
        #MPOS-2基准学习率生成（简化版）
        scheduler.step()
        mydecay = optimizer2.param_groups[0]['lr']/0.1
        print("mydecay is", mydecay)
        newbaselr = bestlr
        # mydecay = 1
        # newbaselr = bestlr*mydecay
        #GWO计算下一epoch学习率
        if dist.get_rank() < GWOsize:
            mylr = abs(GWO(mytensorlist=mytensor_list, populationsize=GWOsize, bestlr=newbaselr, myrank=dist.get_rank(), epoch=epoch, decay = mydecay))
        else:
        #WOA计算下一epoch学习率
            mylr = abs(WOA(mylr=mylr, bestlr=newbaselr, epoch=epoch, mytensorlist=mytensor_list, populationsize = WOAsize, decay = mydecay))
        print("rank ",rank, "epoch", epoch+1,"learning rate is ",mylr)
        #更新优化器
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=mylr,
                            momentum=args.momentum, weight_decay=args.weight_decay)




    log.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.5, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--train_workers", default=8, type=int, help="How many workers for training.")
    args = parser.parse_args()
    size = args.train_workers
    processes = []
    for rank in range(size):
        p = Process(target=main, args=(rank,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
