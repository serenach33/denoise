import os
import math
import torch
import torchaudio
from torchaudio import transforms as T
from torch.utils.data import Dataset
import librosa
from src.utils.data_util import *
from src.utils.biquad import Biquad

class PredictDataProcess(Dataset):
    def __init__(self, args, flag, target_dir):
        self.target_dir = target_dir
        self.args = args
        self.flag = flag
        self.targetsample = self.args.duration * self.args.samplerate

        print('*' * 20)  
        print(f"\nCreating {self.flag} Dataset")
        print(f"Dealing with {len(os.listdir(self.target_dir))} files\n")
        print('*' * 20)
        
        self.all_dataset, self.all_filenames = self.get_dataset()
    
    def get_sample(self, filename):

        filepath = os.path.join(self.target_dir, filename)

        sr = librosa.get_samplerate(filepath)
        audio, _ = torchaudio.load(filepath)

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.args.samplerate:
            resample = T.Resample(sr, self.args.samplerate)
            audio = resample(audio)

        if self.args.use_filter is True:
            
            if 'heart' in self.args.mode:
                lowpass_cutoff = 300
                highpass_cutoff = 50

            if 'lung' in self.args.mode:
                lowpass_cutoff = 1000
                highpass_cutoff = 100

            shelf_gain = 4.5
            Q = 0.707

            bq_lp = Biquad(Biquad.LOWPASS, lowpass_cutoff, self.args.samplerate, Q, 0)
            bq_hp = Biquad(Biquad.HIGHPASS, highpass_cutoff, self.args.samplerate, Q, 0)
            bq_ls = Biquad(Biquad.LOWSHELF, lowpass_cutoff, self.args.samplerate, Q, shelf_gain)
            bq_hs = Biquad(Biquad.HIGHSHELF, highpass_cutoff, self.args.samplerate, Q, shelf_gain)

            audio = biquad_filter(audio.to(self.args.device), bq_lp, bq_hp, bq_ls, bq_hs)       

        return audio

    def get_dataset(self):

        dataset = []
        filenames = []

        files = os.listdir(self.target_dir)

        for i in range(len(files)):
            audio = self.get_sample(files[i])
            if audio.shape[-1] == 0:
                continue
            if audio.shape[-1] > self.targetsample:
                cut_time = audio.shape[-1] // self.targetsample

                for idx in range(cut_time):
                    cut_audio = audio[..., self.targetsample*idx:self.targetsample*(idx+1)]
                    # cut_audio = torch.Tensor(cut_audio).unsqueeze(dim=0)

                    dataset.append((cut_audio, idx))
                    filenames.append(files[i])

                if audio.shape[-1] / cut_time > self.targetsample:
                    last_audio = audio[..., self.targetsample*cut_time:]
                    if self.args.pad == 'zero':
                        tmp = torch.zeros(1, self.targetsample, dtype=torch.float32)
                        diff = self.targetsample - last_audio.shape[-1]
                        tmp[..., diff//2:last_audio.shape[-1]+diff//2] = last_audio
                        pad_data = tmp
            
                    elif self.args.pad == 'repeat':
                        ratio = math.ceil(self.targetsample / last_audio.shape[-1])
                        pad_data = last_audio.repeat(1, ratio)
                        pad_data = pad_data[..., :self.targetsample]

                        if self.args.mode == 'lung':
                            pad_data = self.fade_out(pad_data)
                    
                    dataset.append((pad_data, idx+1))
                    filenames.append(files[i])

            #Just Pad
            else:
                if self.args.pad == 'repeat':
                
                    ratio = math.ceil(self.targetsample / audio.shape[-1])
                    audio = audio.repeat(1, ratio) #dim=(1,ratio)
                    audio = audio[...,:self.targetsample]

                    if 'lung' in self.args.mode:
                        audio = self.fade_out(audio)

                elif self.args.pad == 'zero':
                    tmp = torch.zeros(1, self.targetsample, dtype=torch.float32)
                    diff = self.targetsample - audio.shape[-1]
                    tmp[...,diff//2:audio.shape[-1]+diff//2] = audio

                    audio = tmp

                dataset.append((audio, 0))
                filenames.append(files[i])

        return dataset, filenames

    def __getitem__(self, index):
        
        audio, split_index, filename = self.all_dataset[index][0], self.all_dataset[index][1], self.all_filenames[index]

        return audio, split_index, filename
    
    def __len__(self):
        print("Total Data Count:", len(self.all_dataset))

        return len(self.all_dataset)