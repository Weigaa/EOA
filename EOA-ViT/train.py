# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import math
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size,get_rank
import json

logger = logging.getLogger(__name__)

def write_list_to_json(list, json_file_name, json_file_save_path):
    """
    将list写入到json文件
    :param list:
    :param json_file_name: 写入的json文件名字
    :param json_file_save_path: json文件存储路径
    :return:
    """

    path = json_file_save_path + json_file_name
    #os.chdir(json_file_save_path)
    with open(path, 'a') as f:
        json.dump(list, f)
        f.write('\n')
    f.close()

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
    mydist = ((mydist+4)/8) * 0.01
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
    dist1 = ((dist1 + 4)/8) * 0.01
    dist2 = ((dist2 + 4) / 8) * 0.01
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
    dist1 = ((dist1 + 4) / 8) * 0.01
    dist2 = ((dist2 + 4) / 8) * 0.01
    dist3 = ((dist3 + 4) / 8) * 0.01
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
            dist1 = ((dist1 + 4) / 8) * 0.01
            dist2 = ((dist2 + 4) / 8) * 0.01
            dist3 = ((dist3 + 4) / 8) * 0.01
            omega[i][1] = ((alpha[1] - dist1) + (beta[1] - dist2)+(delta[1] - dist3)) / 3
            return omega[i][1]

#鲸鱼优化算法
def WOA(mylr, bestlr, epoch, mytensorlist, populationsize, decay = 1):
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
        mydist = ((mydist + 1.67) / 4.38) * 0.01
        mylr = bestlr + mydist
        return mylr
    else:
        A = getA(t = epoch)
        if abs(A) < 1:
            D = getD(bestlr,mylr)
            # 更新变化量dist
            mydist = A*D
            # dist进行归一化，以免偏移最优值过大
            mydist = ((mydist + 4) / 8) * 0.01
            mylr = bestlr - mydist
        else:
            randid = random.randint(0,populationsize-1)
            randlr = mytensorlist[randid][1]
            D = getD(randlr, mylr)
            # 更新变化量dist
            mydist = A*D
            # dist进行归一化，以免偏移最优值过大
            mydist = ((mydist + 4) / 8) * 0.01
            mylr = randlr-mydist
        return mylr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 10 if args.dataset == "cifar10" else 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, model):
    """ Train the model """
    rank = get_rank()
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", 'writer_'+str(rank)))

    GWOsize = 4
    WOAsize = get_world_size() - GWOsize

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    print('train_batch_size is ', args.train_batch_size)

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    np.random.seed(rank)
    # 为不同种群创建不同初始学习率
    # 初始为不同节点增加0.9~1.1的扰动
    rand = 1.0 + 0.1 * np.random.rand()
    # GWO使用线性缩放，WOA使用根号缩放
    if rank < GWOsize:
        mylr = args.learning_rate * get_world_size() * rand
    else:
        mylr = args.learning_rate * math.sqrt(get_world_size()) * rand

    # 用来update
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=mylr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # 用来做衰减策略
    optimizer2 = torch.optim.SGD(model.parameters(),
                                lr=3e-2,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, 200, eta_min=0, last_epoch=-1)
    t_total = args.num_steps

    # Distributed training
    if args.local_rank != -1:
        #model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank%2])

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    tensor_list = [torch.zeros(2, dtype=torch.double).to(args.device) for _ in range(get_world_size())]
    mytensor_list = [[0, 0] for _ in range(dist.get_world_size())]


    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    epoch = 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            if (step + 1) % args.gradient_accumulation_steps != 0:
                if rank == -1:
                    batch = tuple(t.to(args.device) for t in batch)
                    x, y = batch
                    loss = model(x, y)
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                else:
                    with model.no_sync():
                    # 因为每次loss.backward（）的时候，torch都会自动gradient all_reduce
                    # 所以为了加速，在没有到达梯度累加循环一轮时最后一次时gradient all_reduce 只需要在这轮的最后一次计算一次就好了
                    # 所以在其他K-1次里，需要把sync 关了， 这样每次loss.backward的时候不会做gradient all_reduce
                        batch = tuple(t.to(args.device) for t in batch)
                        x, y = batch
                        loss = model(x, y)
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        loss.backward()

            else:
                batch = tuple(t.to(args.device) for t in batch)
                x, y = batch
                loss = model(x, y)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                loss.backward()
                losses.update(loss.item() * args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                #if args.local_rank in [-1, 0]:
                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    #writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()

                # if global_step % t_total == 0:
                #     break
                # if global_step >= 2:
                    # break

        # 一个epoch之后 把用于累加多个setp loss 的 losses重置
        loss_val = losses.val
        losses.reset()

        # 一个epoch结束 1.正确率验证 2.选出最好的lr 3.参数归一 4.更新每个rank对应的lr
        # 1.2.MPOS-1选举最优学习率（简化版）
        epoch_accuracy = valid(args, model, writer, test_loader, global_step)
        model.train()
        write_list_to_json([epoch,epoch_accuracy,mylr, loss_val],'rank_'+str(rank)+'.json','')
        if args.local_rank == 0:
            print([epoch,epoch_accuracy,mylr, loss_val])
        
        sharetensor = torch.tensor([epoch_accuracy, mylr]).to(args.device)
        getbesthyperparameter(tensor_list, sharetensor)
        for i in range(len(tensor_list)):
            mytensor_list[i] = tensor_list[i].tolist()
        if rank == 0:
            print(mytensor_list)
        bestrank = mytensor_list.index(max(mytensor_list))
        bestlr = mytensor_list[bestrank][1]

        # 3. 参数归一
        average_params(model)

        # 更新lr
        # MPOS-2基准学习率生成（简化版）
        scheduler.step()
        mydecay = optimizer2.param_groups[0]['lr']/(3e-2)
        # print("mydecay is", mydecay)
        # mydecay = 1
        newbaselr = bestlr
        # GWO计算下一epoch学习率
        if dist.get_rank() < GWOsize:
            mylr = abs(GWO(mytensorlist=mytensor_list, populationsize=GWOsize, bestlr=newbaselr, myrank=dist.get_rank(),
                           epoch=epoch, decay = mydecay))
        else:
            # WOA计算下一epoch学习率
            mylr = abs(
                WOA(mylr=mylr, bestlr=newbaselr, epoch=epoch, mytensorlist=mytensor_list, populationsize=WOAsize, decay = mydecay))
        print("rank ", rank, "epoch", epoch + 1, "learning rate is ", mylr)
        # 更新优化器
        for para in optimizer.param_groups:
            para['lr'] = mylr

        epoch += 1
        if epoch >= args.epochs:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="SGD Momentum.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--epochs", default=200, type=int,
                        help="Total number of epochs.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--train_workers", type=int, default=8,
                        help="How many workers for training.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank % 2)
        device = torch.device("cuda", args.local_rank % 2)
        torch.distributed.init_process_group(backend='gloo', rank = args.local_rank, world_size=args.train_workers,
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    # set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
