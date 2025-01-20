from argparse import ArgumentParser

parser = ArgumentParser()

#seed
parser.add_argument("--seed", type=int, default=1)

#mode
parser.add_argument("--mode", type=str, help='{heart or lung}_{pet or not}_{disease or sound or nab}_{MINE or MG(murmur grade)}')

#tune
parser.add_argument("--tune", action='store_true')

#test
parser.add_argument("--test", action='store_true')
parser.add_argument("--ckpt_path", type=str, help='stored checkpoint path for inference')

#logging
parser.add_argument("--project_name", type=str)
parser.add_argument("--description", type=str)
parser.add_argument("--no_logging", action='store_true')
parser.add_argument("--is_debug", action="store_true")
parser.add_argument("--group", type=str)

#general
parser.add_argument("--num_classes", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--method", type=str, default='ce')
parser.add_argument("--epoch", type=int, default=100)

#model
parser.add_argument("--backbone", type=str, default='cnn6')
parser.add_argument("--dropout", action='store_true')
parser.add_argument("--scratch", action='store_true')
parser.add_argument("--weights_path", type=str, default='weights')
parser.add_argument("--in_channel", type=int, default=1)
parser.add_argument("--mix_beta", type=float, default=1.0)
parser.add_argument("--proj_dim", type=int, default=768)
parser.add_argument("--s_patchout_t", type=int, default=40)
parser.add_argument("--s_patchout_f", type=int, default=4)
parser.add_argument("--stride", type=int, default=10)
parser.add_argument("--patch_size", type=int, default=10)
parser.add_argument('--temperature', type=float, default=0.06)
parser.add_argument('--loss', type=str, help='type of loss')
parser.add_argument("--loss_param", type=str, default='weight', choices=['weight', 'label_smoothing', 'both'])
parser.add_argument('--label_smoothing_rate', type=float, default=0.1)
parser.add_argument('--binary_threshold', type=float, default=0.5)
parser.add_argument("--embed_dim", type=int, default=768)
parser.add_argument("--imagenet_pretrain", action='store_true')
parser.add_argument("--audioset_pretrain", action='store_true')

#optimizer
parser.add_argument("--optimizer", type=str, default='Adam')
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--lr", type=float, default=1e-4)

#Parameter
parser.add_argument("--tau", type=float, default=0.06)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--use_multitask", action='store_true')

#data
parser.add_argument("--use_h5", action="store_true")
parser.add_argument("--split_mode", type=str, default="pid", choices=["patient", "file"])
parser.add_argument("--unpack", action='store_true', help='if unloading tarfile is necessary')
parser.add_argument("--data_dir", type=str, help='top data directory')
parser.add_argument("--tarfile", type=str, help='tarfile includes audio and txt files and metadata.csv')
parser.add_argument("--duration", type=int, default=8)
parser.add_argument("--use_filter", action='store_true')
parser.add_argument("--pad", type=str, default='repeat', choices=['repeat', 'zero'])
parser.add_argument("--samplerate", type=int, default=8000)
parser.add_argument("--nfft", type=int, default=1024)
parser.add_argument("--nmels", type=int, default=80)
parser.add_argument("--win_length", type=int, default=1024)
parser.add_argument("--hop_length", type=int, default=512)
parser.add_argument("--fmin", type=int, default=10, help='heart: 10, lung: 50')
parser.add_argument("--fmax", type=int, default=500, help='heart: 500, lung: 2000')
parser.add_argument("--fade_sample_ratio", type=int, default=16, help='fade in and out for lung')
parser.add_argument("--transform_type", type=str, default='mel', choices=['mel', 'fbank'])
parser.add_argument("--height", type=int, default=798)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--resize", type=int, default=1)
parser.add_argument("--use_normalization", action="store_true")
parser.add_argument("--use_standardization", action="store_true")

# for augmentation
parser.add_argument("--augment_type", type=str, help='choose augmentation method', choices=['Arti', 'Spec', 'None'])
parser.add_argument("--arti_txt_dir", type=str, help='artifact txt files directory')
parser.add_argument("--arti_keep_duration", type=int, default=3, help='longest artifact files duration')
parser.add_argument("--p", type=float, default=0.1)
parser.add_argument("--normal_augment", action='store_true')
parser.add_argument("--artifact_augment", action='store_true')
parser.add_argument("--normal_augment_num", type=int, default=1000, help='number of data to make and add')
parser.add_argument("--artifact_augment_num", type=int, default=200, help='number of data to make and add')

#for Test
parser.add_argument("--pretrained", action='store_true')
parser.add_argument("--pretrained_ckpt", type=str)

#for Predict
parser.add_argument("--predict", action='store_true')

#save
parser.add_argument("--save_dir", type=str, default='/work/chs/school/paper/denoise/save')

#reload to train
parser.add_argument("--purpose", type=str)