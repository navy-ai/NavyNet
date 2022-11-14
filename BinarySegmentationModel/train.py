import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import change_detection_pytorch as cdp
from change_detection_pytorch.ufpn.model import UFPNnet
from change_detection_pytorch.datasets.PRCV_CD import PRCV_CD_Dataset
import wandb

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    wandb
    wandb.init(project='BSM')

    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_torch(seed=42)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)

    model = UFPNnet(
    encoder_name="timm-regnety_320",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights='imagenet',  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,  # model output channels (number of classes in your datasets)
    decoder_attention_type="eca",
    siam_encoder=True,
    decoder_channels=[384, 256, 128, 64, 32],
    fusion_form='diff',
    head='cond',
    )
    
    train_dataset = PRCV_CD_Dataset('../train/',
                                    sub_dir_1='input1',
                                    sub_dir_2='input2',
                                    img_suffix='.png',
                                    ann_dir='../train/binary_mask',
                                    size=256,
                                    debug=False)

    valid_dataset = PRCV_CD_Dataset('../train/val',
                                    sub_dir_1='input1',
                                    sub_dir_2='input2',
                                    img_suffix='.png',
                                    ann_dir='../train/val/binary_mask',
                                    size=256,
                                    debug=False,
                                    test_mode=True)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)


    ce_weight = torch.tensor([1.0, 2.0]).to(DEVICE)
    loss = cdp.utils.losses.MultiHeadCELoss(weight=ce_weight, loss2=True, loss2_weight=1.0)

    # loss1 = cdp.utils.losses.CrossEntropyLoss(weight=ce_weight)
    # loss2 = cdp.losses.DiceLoss(mode='multiclass')
    # loss = cdp.losses.HybridLoss(loss1, loss2)

    metrics = [
        cdp.utils.metrics.Fscore(activation='argmax2d'),
        # cdp.utils.metrics.Recall(activation='argmax2d'),
        cdp.utils.metrics.Binary_mIOU(activation='argmax2d'),
        cdp.utils.metrics.IOU0(activation='argmax2d'),
        cdp.utils.metrics.IOU1(activation='argmax2d'),
        cdp.utils.metrics.Accuracy(activation='argmax2d'),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45, 60, 70, 80, 90, 96], gamma=0.5)

    # create epoch runners
    train_epoch = cdp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
        is_train = True,
        # accumulation=True,
        # scale=(0.85, 1.15),
    )

    valid_epoch = cdp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
        is_train = False,
        # TTA=True,
    )

    max_score = 0
    MAX_EPOCH = 100
    JSON_LOG = []

    for i in range(1, MAX_EPOCH + 1):

        print('\nEpoch: {}'.format(i))
        if MAX_EPOCH - i <= 10:
            train_logs = train_epoch.run(train_loader)
        else:
            train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        scheduler_steplr.step()
        
        print('train_logs ', train_logs)
        print('valid_logs ', valid_logs)
        JSON_LOG.append({'epoch': i, 'train_logs': train_logs, 'valid_logs': valid_logs})

        torch.save(model, './save/epoch_'+str(i)+'.pth')
        print('Model saved!')


    with open('log' + '.json', 'w') as fout:
        json.dump(JSON_LOG, fout, indent=2)

    valid_epoch.infer_vis(valid_loader, slide=False, image_size=512, window_size=256,
                                    save_dir='./res', evaluate=True, suffix='.png')

    wandb.finish()
