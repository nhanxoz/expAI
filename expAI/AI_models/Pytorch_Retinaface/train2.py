import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from expAI.AI_models.Pytorch_Retinaface.data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from expAI.AI_models.Pytorch_Retinaface.layers.modules import MultiBoxLoss
from expAI.AI_models.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from expAI.AI_models.Pytorch_Retinaface.models.retinaface import RetinaFace

import time
import datetime
import math
import json

from expAI.models import *


args_network = 'mobile0.25'
args_num_workers = 1
args_lr = float(1e-3)
args_momentum = float(0.9)
args_resume_net = None
args_resume_epoch = 0
args_weight_decay = float(5e-4)
args_gamma = float(0.1)
args_save_folder = './expAI/AI_models/Pytorch_Retinaface/weights/'

# json_config = {
#     'name': 'mobilenet0.25',
#     'save_folder': './AI_models/Pytorch_Retinaface/weights/',
#     'gpu_train': True,
#     'batch_size': 32,
#     'ngpu': 1,
#     'epoch': 250,
#     'image_size': 640
# }
def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    initial_lr = args_lr
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_mnet(para_id,dataset_path,json_config):
    dataset_path = './datasets/'+dataset_path+'/label.txt'
    json_config = json.loads(json_config)
    

    print('dataset_path',dataset_path)
    print('json_config',json_config)
    print('pre_id',para_id)
    if not os.path.exists(json_config['save_folder']):
        os.mkdir(json_config['save_folder'])
    cfg = cfg_mnet
    rgb_mean = (104, 117, 123) # bgr order
    num_classes = 2
    img_dim = json_config['image_size']
    num_gpu = json_config['ngpu']
    batch_size = json_config['batch_size']
    max_epoch = json_config['epoch']
    gpu_train = json_config['gpu_train']
    num_workers = args_num_workers
    momentum = args_momentum
    weight_decay = args_weight_decay
    initial_lr = args_lr
    gamma = args_gamma
    training_dataset = dataset_path
    save_folder = args_save_folder
    net = RetinaFace(cfg=cfg)
    print("Printing net...")
    print(net)
    if num_gpu > 1 and gpu_train:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()
    cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()
    net.train()
    epoch = 0 + args_resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection( training_dataset,preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args_resume_epoch > 0:
        start_iter = args_resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
            epoch += 1


        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))


        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

        _para = Paramsconfigs.objects.get(pk=para_id)
        if _para.trainningstatus == 0:
            _new_result = Trainningresults()
            _new_result.configid = _para
            _new_result.accuracy = 0
            _new_result.lossvalue = loss
            _new_result.trainresultindex = iteration+1
            _new_result.is_last = True
            _new_result.save()
            return

        else:
            _new_result = Trainningresults()
            _new_result.configid = _para
            _new_result.accuracy = 0
            _new_result.lossvalue = loss
            _new_result.trainresultindex = iteration+1
            _new_result.is_last = False
            _new_result.save()

    _para = Paramsconfigs.objects.get(pk=para_id)
    _para.trainningstatus = 0
    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')

    







def train_resnet(pre_id,dataset_path,json_config):
    print('dataset_path',dataset_path)
    print('json_config',json_config)
    print('pre_id',pre_id)    