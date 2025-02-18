import os
import random
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
from src.models.resnet import ResNet38
from src.models.ast_models import ASTModel
from src.models.cnn6 import CNN6
from src.models.method import CE

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)


def get_model(args):
    kwargs = {}

    if args.backbone == 'cnn6':
    
        kwargs['num_classes'] = args.num_classes
        kwargs['do_dropout'] = args.dropout
        kwargs['from_scratch'] = args.scratch
        kwargs['path_to_weight'] = os.path.join(args.weights_path, 'Cnn6_mAP=0.343.pth')
        kwargs['in_channel'] = args.in_channel

        loaded_model = CNN6(**kwargs)

    if args.backbone == 'resnet38':

        kwargs['num_classes'] = args.num_classes
        kwargs['do_dropout'] = args.dropout
        
        loaded_model = ResNet38(**kwargs)

    if args.backbone == "ast":

        kwargs['label_dim'] = args.num_classes # number of total classes
        kwargs['input_fdim'] = args.nmels
        kwargs['input_tdim'] = int(np.ceil((args.duration * args.samplerate) / args.hop_length)) + 1
        kwargs['imagenet_pretrain'] = args.imagenet_pretrain
        kwargs['audioset_pretrain'] = args.audioset_pretrain
        
        loaded_model = ASTModel(**kwargs)

    return loaded_model

def get_method(args, wav_dir, model):
    kwargs = {}
    height = 1
    default_criterion = get_default_criterion(args)
    kwargs['optimizer'] = args.optimizer
    kwargs['lr'] = args.lr
    kwargs['wd'] = args.wd
    kwargs['num_classes'] = args.num_classes
    kwargs['transform_type'] = args.transform_type
    kwargs['augment_type'] = args.augment_type
    kwargs["use_standardization"] = args.use_standardization
    kwargs["use_normalization"] = args.use_normalization
    kwargs["batch"] = args.batch_size    
    kwargs['mode'] = args.mode

    # if args.purpose == 'val_reload':
    #     kwargs['fig_sav_dir'] = args.save_dir
    # elif args.ckpt_path == None:
    #     kwargs['fig_sav_dir'] = args.save_dir
    # else:
    #     kwargs['fig_sav_dir'] = args.ckpt_path[:args.ckpt_path.rfind('/')]
    # kwargs['purpose'] = args.purpose
    if args.augment_type == 'Arti':
        kwargs['augment_dict'] = {
            'arti_txt_dir' : args.arti_txt_dir,
            'wav_dir' : wav_dir,
            'keep_duration' : args.arti_keep_duration,
            'desired_duration' : args.duration,
            'samplerate' : args.samplerate,
            'p' : args.p
        }
    else:
        kwargs['augment_dict'] = None
    
    if args.transform_type == 'fbank':
        # width = 798
        # height = args.nmels
        width = args.nmels        
        if args.duration == 8:
            height = 798
    else:
        #mel spectrogram shape (nmels, num_frames)
        num_frames = int(np.ceil((args.duration * args.samplerate) / args.hop_length)) + 1
        width = num_frames
        height = args.nmels

    if args.use_resize:
        kwargs['transform_dict'] = {
            'samplerate' : args.samplerate,
            'nfft' : args.nfft,
            'nmels' : args.nmels,
            'win_length' : args.win_length,
            'hop_length' : args.hop_length,
            'fmin' : args.fmin,
            'fmax' : args.fmax,
            'img_size' : (224, 224)
        }
    else:
        kwargs['transform_dict'] = {
            'samplerate' : args.samplerate,
            'nfft' : args.nfft,
            'nmels' : args.nmels,
            'win_length' : args.win_length,
            'hop_length' : args.hop_length,
            'fmin' : args.fmin,
            'fmax' : args.fmax,
            'img_size' : (int(height * args.resize), int(width * args.resize))
        }


    # if args.method == 'ce':
    #     kwargs['criterion'] = [default_criterion]
    #     method = CE(**kwargs, custom_model=model)

    # if args.method == 'patchmix':
    #     kwargs['criterion'] = [default_criterion, PatchMixLoss(criterion=default_criterion)]
    #     method = Patchmix(**kwargs, custom_model=model)

    # if args.method == 'patchmix_cl':
    #     kwargs['criterion'] = [default_criterion, PatchMixConLoss(temperature=args.temperature)]
    #     method = Patchmix_Cl(**kwargs, custom_model=model)
    kwargs['criterion'] = [default_criterion]
    method = CE(**kwargs, custom_model=model)
    return method

def get_default_criterion(args):
 
    #for heart_pet_disease only
    #CLASS WEIGHT : tensor([0.2936, 0.1710, 0.1743, 0.1593, 0.2019])
    weights = torch.tensor([3820.0, 3536.0])
    
    weights = 1.0 / (weights / weights.sum())
    weights /= weights.sum()
    print("WEIGHTS ARE", weights)
    
    if args.loss_param == 'weight':
        default_criterion = nn.CrossEntropyLoss(weight=weights)
    if args.loss_param == 'label_smoothing':
        default_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_rate)
    if args.loss_param == 'both':
        default_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing_rate)

    return default_criterion