import os
import tarfile
import torchaudio
import pandas as pd
from torchaudio import transforms as T
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import torch

def extract_tar(tar_file, destination):
    with tarfile.open(tar_file, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile():
                member.name = os.path.basename(member.name)
                tar.extract(member, path=destination)

def get_pth_path(target_dir, args, split):
    default_path = target_dir.split('origin')[0]
    do_filter = args.use_filter
    if do_filter:
        pth_path = os.path.join(default_path, f"{str(args.mode)}_{str(split)}_{str(args.duration)}sec_filter.pth")
    else:
        pth_path = os.path.join(default_path, f"{str(args.mode)}_{str(split)}_{str(args.duration)}sec.pth")
    
    return pth_path

def biquad_filter(data, bq_heart_lp, bq_heart_hp, bq_heart_ls, bq_heart_hs):

    # lowpass
    y = torchaudio.functional.biquad(data, bq_heart_lp.b0, bq_heart_lp.b1, bq_heart_lp.b2,
                                     1., bq_heart_lp.a1, bq_heart_lp.a2)
    y = torchaudio.functional.biquad(y, bq_heart_lp.b0, bq_heart_lp.b1, bq_heart_lp.b2,
                                     1., bq_heart_lp.a1, bq_heart_lp.a2)
    y = torchaudio.functional.biquad(y, bq_heart_lp.b0, bq_heart_lp.b1, bq_heart_lp.b2,
                                     1., bq_heart_lp.a1, bq_heart_lp.a2)
    # highpass
    y = torchaudio.functional.biquad(y, bq_heart_hp.b0, bq_heart_hp.b1, bq_heart_hp.b2,
                                     1., bq_heart_hp.a1, bq_heart_hp.a2)
    y = torchaudio.functional.biquad(y, bq_heart_hp.b0, bq_heart_hp.b1, bq_heart_hp.b2,
                                     1., bq_heart_hp.a1, bq_heart_hp.a2)
    y = torchaudio.functional.biquad(y, bq_heart_hp.b0, bq_heart_hp.b1, bq_heart_hp.b2,
                                     1., bq_heart_hp.a1, bq_heart_hp.a2)
    # lowshelf
    y = torchaudio.functional.biquad(y, bq_heart_ls.b0, bq_heart_ls.b1, bq_heart_ls.b2,
                                     1., bq_heart_ls.a1, bq_heart_ls.a2)
    # highshelf
    y = torchaudio.functional.biquad(y, bq_heart_hs.b0, bq_heart_hs.b1, bq_heart_hs.b2,
                                     1., bq_heart_hs.a1, bq_heart_hs.a2)

    return y

def get_annotation(target_dir, filename, mode):
    if "nab" in mode:
        annotation = pd.read_csv(os.path.join(target_dir, filename+'.txt'),
                                names=['start', 'end', 'normal', 'abnormal', 'artiract'], delimiter='\t')

    return annotation

def get_data_info(row, mode):
    
    onset = row['start']
    offset = row['end']
    labels = row.keys()[2:].values.tolist()

    #one hot
    for i in range(len(labels)):
        labels[i] = row[f"{labels[i]}"]
    
    #lung only (for both crackle and wheeze)
    if labels.count(1) > 1:
        if mode != 'lung':
            print('error : multiple label')
        label = 4
    else:
        label = labels.index(1)

    return onset, offset, label

class Get_MelSpec(pl.LightningModule):
    def __init__(self, transform_dict):
        super().__init__()
        self.melspec = T.MelSpectrogram(sample_rate=transform_dict['samplerate'],
                                n_fft=transform_dict['nfft'],
                                n_mels=transform_dict['nmels'],
                                win_length=transform_dict['win_length'],
                                hop_length=transform_dict['hop_length'],
                                f_min=transform_dict['fmin'],
                                f_max=transform_dict['fmax'],
                                )
        self.amplitude_to_db = T.AmplitudeToDB()
    
    def forward(self, x):
        
        out = self.melspec(x)
        out = self.amplitude_to_db(out)

        mean, std =  -4.2677393, 4.5689974
        out = (out - mean) / (std * 2)

        return out
    
class Get_Fbank(pl.LightningModule):
    def __init__(self, transform_dict):
        super().__init__()
        self.samplerate = transform_dict['samplerate']
        self.nmels = transform_dict['nmels']
        self.img_size = transform_dict['img_size']
        self.resizer = ResizeTensor(target_size=self.img_size)
    
    def forward(self, x):
        
        fbanks = []
        for i in range(len(x)):
            audio = x[i]
            fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True,
                                                sample_frequency=self.samplerate,
                                                use_energy=False,
                                                window_type='hanning',
                                                num_mel_bins=self.nmels,
                                                dither=0.0,
                                                frame_shift=10)
            
            mean, std =  -4.2677393, 4.5689974 #ast 논문 값
            fbank = (fbank - mean) / (std * 2) # mean / std
            # Step 1: Rotate counterclockwise (90 degrees)
            fbank = torch.rot90(fbank, k=1, dims=(0, 1))
            
            # fbank = fbank.T
            fbank = self.resizer(fbank)

            fbanks.append(fbank)
        
        fbanked = torch.stack(fbanks).unsqueeze(dim=1)

        return fbanked
    
class ResizeTensor:
    def __init__(self, target_size=(224, 224)):
        self.target_height, self.target_width = target_size  # (224, 224)

    def __call__(self, tensor):
        original_height, original_width = tensor.shape  # (768, 64)

        # Step 1: Split into chunks of height 224
        chunks = []
        for start in range(0, tensor.shape[1], self.target_width):
            chunk = tensor[:, start:start + self.target_width]  # Take width slice

            if chunk.shape[1] < self.target_width:
                # Instead of padding, repeat the chunk until width = 224
                repeat_factor = (self.target_width // chunk.shape[1]) + 1  # Calculate repetitions
                chunk = chunk.repeat(1, repeat_factor)  # Repeat along width axis
                chunk = chunk[:, :self.target_width]  # Trim to exactly 224 width

            chunks.append(chunk.detach().cpu())

        # Step 3: Stack chunks vertically
        stacked_tensor = torch.cat(chunks, dim=0)  # Concatenating along height axis

        # Step 4: Resize height to match 224 (if needed)
        if stacked_tensor.shape[0] != self.target_height:
            stacked_tensor = F.interpolate(
                stacked_tensor.unsqueeze(0).unsqueeze(0),  # Add batch & channel dims
                size=(self.target_height, self.target_width), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)  # Remove batch & channel dims
            stacked_tensor = stacked_tensor.to(dtype=torch.float16).to(device='cuda:0')
        return stacked_tensor

# class Get_Fbank(pl.LightningModule):
#     def __init__(self, transform_dict):
#         super().__init__()
#         self.samplerate = transform_dict['samplerate']
#         self.nmels = transform_dict['nmels']
#         self.img_size = transform_dict['img_size']
    
#     def forward(self, x):
        
#         fbanks = []
#         for i in range(len(x)):
#             audio = x[i]
#             fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True,
#                                                 sample_frequency=self.samplerate,
#                                                 use_energy=False,
#                                                 window_type='hanning',
#                                                 num_mel_bins=self.nmels,
#                                                 dither=0.0,
#                                                 frame_shift=10)
            
#             mean, std =  -4.2677393, 4.5689974 #ast 논문 값
#             fbank = (fbank - mean) / (std * 2) # mean / std

#             fbanks.append(fbank)
        
#         fbanked = torch.stack(fbanks).unsqueeze(dim=1)

#         return fbanked
    
class Normalization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.power_to_db = T.AmplitudeToDB()

    def forward(self, x):

        x = self.power_to_db(x)  # x shape = [batch, 1, nmels, time]
        # if we do x.min() and x.max() it only gets the batch's last data's min and max
        # x.size(0) is the batch size, x_t shape = [batch, nmels*time]
        x_t = x.reshape(x.size(0), -1)
        # min value for each batch, shape = [batch, 1]
        min = x_t.min(1, keepdim=True)[0]
        # max value for each batch, shape = [batch, 1]
        max = x_t.max(1, keepdim=True)[0]

        # print(f"max {max} min {min}")

        epsilon = 1e-8
        x_t = (x_t - min) / (max - min + epsilon)
        # x_t = x_t/max
        # x_t reshape to original x shape == [batch, 1, nmels, time]
        return x_t.reshape(x.shape)

class Standardization(torch.nn.Module):
    def __init__(self, melspec=None, train_loader=None,  mean=None, std=None):
        super().__init__()

        

        if train_loader is None:
            self.mean = mean
            self.std = std
        else:
           #mean: -15.21766185760498, std: 16.169034957885742 =========================

            # normalize = Normalization()
            # melspec = torch.nn.Sequential(melspec, normalize)

            if melspec is not None:
                melspec = torch.nn.Sequential(melspec)

            mean = []
            std = []
            for i, (audio, label, _, _) in enumerate(train_loader):
                
                if melspec is not None:
                    audio = melspec(audio)
                cur_mean = torch.mean(audio)
                cur_std = torch.std(audio)
                mean.append(cur_mean)
                std.append(cur_std)

            self.mean = torch.mean(torch.stack(mean))
            self.std = torch.mean(torch.stack(std))

        
            
            print(f"mean: {self.mean}, std: {self.std} =========================")

    def forward(self, x):

        return (x-self.mean) / self.std