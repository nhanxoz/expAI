from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler,SequentialSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import numpy as np
import os
from expAI.models import *
from expAI.AI_models.Pytorch_FaceNet import facenet_src_lfw
import cv2
from PIL import Image,ImageDraw
import json


def train(para_id,dataset_path,json_config):

    json_config = json.loads(json_config)

    data_dir = './datasets/' + dataset_path

    try:
        epochs = json_config['epochs']
    except:
        epochs = 10
    batch_size = 16
    print('epoch',epochs)
    print('batch_size',batch_size)
    print(data_dir)


    workers = 0 if os.name == 'nt' else 8
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))    

    mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device)

    dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
    dataset.samples = [
        (p, p.replace(data_dir, data_dir + '_cropped'))
            for p, _ in dataset.samples
    ]
            
    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )

    for i, (x, y) in enumerate(loader):
        mtcnn(x, save_path=y)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
        
    # Remove mtcnn to reduce GPU memory usage
    del mtcnn
    resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=len(dataset.class_to_idx)
    ).to(device)
    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, [5, 10])

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
    img_inds = np.arange(len(dataset))
    np.random.shuffle(img_inds)
    train_inds = img_inds
    #train_inds = img_inds[:int(0.8 * len(img_inds))]
    #val_inds = img_inds[int(0.8 * len(img_inds)):]

    train_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_inds)
    )
    # val_loader = DataLoader(
    #     dataset,
    #     num_workers=workers,
    #     batch_size=batch_size,
    #     sampler=SubsetRandomSampler(val_inds)
    # )
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }

    print('\n\nInitial')
    print('-' * 10)
    resnet.eval() 
    training.pass_epoch(
        resnet, loss_fn, train_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=None
    )

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        resnet.train()
        i_loss,metric = training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=None
        )

        # resnet.eval()
        # training.pass_epoch(
        #     resnet, loss_fn, val_loader,
        #     batch_metrics=metrics, show_running=True, device=device,
        #     writer=writer
        # )

        _para = Paramsconfigs.objects.get(pk=para_id)
        if _para.trainningstatus == 0 or epoch+1 == epochs:
            _new_result = Trainningresults()
            _new_result.configid = _para
            _new_result.accuracy = float(metric.get('acc'))
            _new_result.lossvalue = float(i_loss)
            _new_result.trainresultindex = epoch+1
            _new_result.is_last = True
            _new_result.save()
            break

        else:
            _new_result = Trainningresults()
            _new_result.configid = _para
            _new_result.accuracy = float(metric.get('acc'))
            _new_result.lossvalue = float(i_loss)
            _new_result.trainresultindex = epoch+1
            _new_result.is_last = False  
            _new_result.save()
    
    _para = Paramsconfigs.objects.get(pk=para_id)
    save_model_path = './weights/' + str(_para.pk)
    if not os.path.exists(save_model_path):
        # Create a new directory because it does not exist
        os.makedirs(save_model_path)
    _para.trainningstatus = 0
    _para.configaftertrainmodelpath = save_model_path + '/facenet.pth'
    _para.save()
    torch.save(resnet,_para.configaftertrainmodelpath)
    


def test(ressult_id,dataset_path):

    data_dir = 'datasets/'+ dataset_path+'/lfw'
    pairs_path = 'datasets/'+ dataset_path + '/pairs.txt'

    batch_size = 16
    epochs = 15
    workers = 0 if os.name == 'nt' else 8
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    mtcnn = MTCNN(
        image_size=160,
        margin=14,
        device=device,
        selection_method='center_weighted_size'
    )
    orig_img_ds = datasets.ImageFolder(data_dir, transform=None)
    # overwrites class labels in dataset with path so path can be used for saving output in mtcnn batches
    orig_img_ds.samples = [
        (p, p)
        for p, _ in orig_img_ds.samples
    ]

    loader = DataLoader(
        orig_img_ds,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )
    crop_paths = []
    box_probs = []

    for i, (x, b_paths) in enumerate(loader):
        crops = [p.replace(data_dir, data_dir + '_cropped') for p in b_paths]
        mtcnn(x, save_path=crops)
        crop_paths.extend(crops)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
    # Remove mtcnn to reduce GPU memory usage
    del mtcnn
    torch.cuda.empty_cache()
    # create dataset and data loaders from cropped images output from MTCNN

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)

    embed_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )
    
    # Load pretrained resnet model
    resnet = InceptionResnetV1(
        classify=False,
        pretrained='vggface2'
    ).to(device)
    classes = []
    embeddings = []
    resnet.eval()
    with torch.no_grad():
        for xb, yb in embed_loader:
            xb = xb.to(device)
            b_embeddings = resnet(xb)
            b_embeddings = b_embeddings.to('cpu').numpy()
            classes.extend(yb.numpy())
            embeddings.extend(b_embeddings)
    embeddings_dict = dict(zip(crop_paths,embeddings))
    pairs = facenet_src_lfw.read_pairs(pairs_path)
    path_list, issame_list = facenet_src_lfw.get_paths(data_dir+'_cropped', pairs)
    embeddings = np.array([embeddings_dict[path] for path in path_list])

    tpr, fpr, accuracy, val, val_std, far, fp, fn = facenet_src_lfw.evaluate(embeddings, issame_list)

    print(accuracy)
    
    _ressult = Results.objects.get(pk = ressult_id)
    _ressult.resultaccuracy = np.mean(accuracy)
    _ressult.save()


from sklearn.model_selection import KFold
from scipy import interpolate
import numpy as np

# LFW functions taken from David Sandberg's FaceNet implementation
def cosine_distance(emb1,emb2):
    A = np.array(emb1)
    B = np.array(emb2)

    cosine = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
    return cosine



def predict(pre_id,trained_model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device, keep_all=True
    )
    model = torch.load(trained_model)
    model.eval()

    _pre = Predict.objects.get(pk=pre_id)
    result_path = str(_pre.inputpath)[:8] + 'predict_result' + str(_pre.inputpath)[20:]
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    _pre.outputpath = result_path
    _pre.save()

    person_embs = []

    for img_name in os.listdir(str(_pre.inputpath)):
        img_name = img_name
        img_raw = Image.open(os.path.join(str(_pre.inputpath),img_name))
        img_cropped = mtcnn(img_raw)
        img_embedding = model(img_cropped)
        img_embedding = img_embedding.to('cpu').detach().numpy()
        person_embs.append(img_embedding[0])
    



    if _pre.datatype == 'image':
        for img_name in os.listdir(str(_pre.inputpath2)):
            img_raw = Image.open(os.path.join(str(_pre.inputpath2),img_name))
            imgs_cropped = mtcnn(img_raw)
            imgs_embedding = model(imgs_cropped)
            imgs_embedding = imgs_embedding.to('cpu').detach().numpy()

            boxes, probs, points = mtcnn.detect(img_raw, landmarks=True)
            img_draw = img_raw.copy()
            draw = ImageDraw.Draw(img_draw)
            for i, (box, point,face_embedding) in enumerate(zip(boxes, points,imgs_embedding)):
                for person_emb in person_embs:
                    distance = cosine_distance(person_emb,face_embedding)
                    if distance > 0.8:
                        draw.rectangle(box.tolist(), width=2,outline ="red")
                        draw.text((box.tolist()[0],box.tolist()[1]),str(distance))
                    else:
                        draw.rectangle(box.tolist(), width=2)
                        draw.text((box.tolist()[0],box.tolist()[1]),str(distance))
            img_draw.save(os.path.join(result_path,img_name))

    elif _pre.datatype == 'video':
        # _pre.outputpath = result_path
        # _pre.save()
        return



