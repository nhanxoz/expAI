import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from expAI.AI_models.Pytorch_Retinaface.data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from expAI.AI_models.Pytorch_Retinaface.layers.modules import MultiBoxLoss
from expAI.AI_models.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from expAI.AI_models.Pytorch_Retinaface.models.retinaface import RetinaFace
from expAI.AI_models.Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from expAI.AI_models.Pytorch_Retinaface.utils.timer import Timer
from expAI.AI_models.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from expAI.AI_models.Pytorch_Retinaface.widerface_evaluate.evaluation import evaluation as cacular_mAP

import time
import datetime
import math
import json

from expAI.models import *
import cv2
import numpy as np




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
    initial_lr = float(1e-3)
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(para_id,dataset_path,json_config):
    args_network = 'mobile0.25'
    args_num_workers = 1
    args_lr = float(1e-3)
    args_momentum = float(0.9)
    args_resume_net = None
    args_resume_epoch = 0
    args_weight_decay = float(5e-4)
    args_gamma = float(0.1)
    args_save_folder = './expAI/AI_models/Pytorch_Retinaface/weights/'
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
    if gpu_train:
        if not torch.cuda.is_available():
            gpu_train = False
    if gpu_train:
        net = net.cuda()
    else:
        net = net.cpu()
    cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        if gpu_train:
            priors = priors.cuda()
        else:
            priors = priors.cpu()
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

    batch_iterator = None
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
        if gpu_train:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
        else:
            images = images.cpu()
            targets = [anno.cpu() for anno in targets]

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
        if _para.trainningstatus == 0 or (iteration+1) == max_iter:
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

def test(result_id,dataset_path):
    print("vafo tesst")
    print(result_id)
    print(dataset_path)
    _ressult = Results.objects.get(pk = result_id)
    _para = Paramsconfigs.objects.get(pk=_ressult.resultconfigid.pk)
    trained_model = str(_para.configaftertrainmodelpath)

    keep_top_k = 750


    cfg = cfg_mnet
    net = RetinaFace(cfg=cfg, phase = 'test')
    if torch.cuda.is_available():
        net = load_model(net,trained_model, False)
    else:
        net = load_model(net,trained_model, True)
    net.eval()
    cudnn.benchmark = True
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    net = net.to(device)

    data = Datasets.objects.get(pk = _ressult.resulttestingdataset.pk)
    test_data_path = './datasets/' + str(data.datasetfolderurl)
    # testing dataset
    testset_folder = test_data_path + '/images/'
    testset_list = testset_folder[:-7] + "wider_val.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if True:
            resize = 1
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > 0.02)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()

        # --------------------------------------------------------------------
        save_folder = test_data_path + '/widerface_evaluate/widerface_txt/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_name = save_folder + img_name[:-4] + ".txt"
        print(save_name)
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        print(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))
    aps = cacular_mAP(test_data_path + '/widerface_evaluate/widerface_txt/',test_data_path + '/ground_truth/')
    _ressult.resultaccuracy = aps[1]
    detail_ap = {}
    detail_ap['easy ap'] = aps[0]
    detail_ap['medium ap'] = aps[1]
    detail_ap['hard ap'] = aps[2]
    _ressult.resultdetail = json.dumps(detail_ap)
    _ressult.save()





def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys))) 
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def predict(pre_id,trained_model):
    import cv2
    import numpy as np
    keep_top_k = 750


    cfg = cfg_mnet
    net = RetinaFace(cfg=cfg, phase = 'test')
    if torch.cuda.is_available():
        net = load_model(net,trained_model, False)
    else:
        net = load_model(net,trained_model,True)
    net.eval()
    cudnn.benchmark = True
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    net = net.to(device)


    _pre = Predict.objects.get(pk=pre_id)
    result_path = str(_pre.inputpath)[:8] + 'predict_result' + str(_pre.inputpath)[20:]
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    _pre.outputpath = result_path
    _pre.save()
    

    print('trained_model',trained_model)
    print('result_path',result_path)
    print('inputpath',_pre.inputpath)
    print(len(os.listdir(str(_pre.inputpath))))
    

    if _pre.datatype == 'image':
        for img_name in os.listdir(str(_pre.inputpath)):
            img_name = img_name
            img_raw = cv2.imread(os.path.join(str(_pre.inputpath),img_name))
            if img_raw is None:
                continue
            
            img = np.float32(img_raw)


            target_size = 1600
            max_size = 2150
            im_shape = img.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            resize = float(target_size) / float(im_size_min)

            if np.round(resize * im_size_max) > max_size:
              resize = float(max_size) / float(im_size_max)
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            tic = time.time()
            loc, conf, landms = net(img)  # forward pass
            print('net forward time: {:.4f}'.format(time.time() - tic))

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > 0.4)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:5000]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, 0.4)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:keep_top_k, :]
            landms = landms[:750, :]

            dets = np.concatenate((dets, landms), axis=1)
            for b in dets:
                # if b[4] < 0.2:
                #     continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image
            cv2.imwrite(os.path.join(result_path,img_name),img_raw)
        # _pre.outputpath = result_path
        # _pre.save()
    elif _pre.datatype == 'video':
        # _pre.outputpath = result_path
        # _pre.save()
        return
        # for filename in os.listdir(str(_pre.inputpath)):
        #     video = cv2.VideoCapture(os.path.join(str(_pre.inputpath),filename))
        #     if video.isOpened() == False:
        #         _pre.details = "Error reading video file"
        #         return
        #     frame_width = int(video.get(3))
        #     frame_height = int(video.get(4))

        #     size = (frame_width, frame_height)
        #     result = cv2.VideoWriter(os.path.join(result_path,filename),
        #                  cv2.VideoWriter_fourcc(*'MJPG'),
        #                  10, size)