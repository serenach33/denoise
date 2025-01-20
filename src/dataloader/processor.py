import os
import math
import tqdm
import torch
import torchaudio
from torchaudio import transforms as T
from torch.utils.data import Dataset
import librosa
import numpy as np
from src.utils.data_util import *
from src.utils.biquad import Biquad
import h5py


class DataProcess(Dataset):
    def __init__(self, args, flag, target_dir, unlabeled=False, is_debug=False):

        self.target_dir = target_dir
        self.args = args
        self.targetsample = self.args.duration * self.args.samplerate
        self.unlabeled = unlabeled
        self.is_debug = is_debug

        self.flag = flag

        if is_debug:
            filepaths = os.listdir(self.target_dir)[:16]
        else:
            filepaths = os.listdir(self.target_dir)

        filenames = set(
            [f.strip().split(".")[0] for f in filepaths if ".wav" in f or ".txt" in f]
        )
        filenames = sorted(filenames)
        foldfile = os.path.join(target_dir, "patient_list_foldwise.txt")

        if self.unlabeled == False:
            patient_dict = {}
            all_patients = open(foldfile).read().splitlines()

            for line in all_patients:

                pid, fold = line.strip().split(" ")
                if flag == "train" and fold == "200":
                    # if train_flag and fold == '200':
                    patient_dict[pid] = fold

                elif flag == "val" and fold == "999":
                    # elif not train_flag and fold == '999':
                    patient_dict[pid] = fold

                elif flag == "test" and fold == "1000":
                    patient_dict[pid] = fold

            print("*" * 20)
            print(
                "\nNumber of Patients in {} dataset: {}\n".format(
                    flag, len(patient_dict)
                )
            )

        self.filenames = []

        for f in filenames:
            try:
                if args.split_mode == "patient":
                    pid = f.split("_")[0].split("-")[1]
                    
                    if pid in patient_dict:
                        self.filenames.append(f)
                else:
                    if f in patient_dict:
                        self.filenames.append(f)  
            except Exception as e:
                pass

        self.pth_path = get_pth_path(self.target_dir, self.args, flag)
        self.label_list = []

        if os.path.exists(self.pth_path):
            print("*" * 20)
            print(f"Loading {flag} dataset...")

            if self.args.use_h5 == False:

                if self.unlabeled == False:
                    pth_dataset = torch.load(self.pth_path)
                    (
                        self.data_list,
                        self.label_list,
                        self.filename_list,
                        self.split_index_list,
                    ) = (
                        pth_dataset["data"],
                        pth_dataset["label"],
                        pth_dataset["file_name"],
                        pth_dataset["split_idx"],
                    )
                else:
                    pth_dataset = torch.load(self.pth_path)
                    (
                        self.data_list,
                        self.filename_list,
                        self.split_index_list,
                    ) = (
                        pth_dataset["data"],
                        pth_dataset["file_name"],
                        pth_dataset["split_idx"],
                    )
            else:
                with h5py.File(self.pth_path, "r") as hf:
                    # dataset의 개수 얻기
                    self.h5_length = int(hf["info"]["file_num"][()])

                    self.class_nums = hf["info"]["class_num"][:]

                    print(
                        f"1.---------------------{self.flag} {self.h5_length} ---------------------"
                    )

            print("*" * 20)
            print(f"Loaded {flag} dataset!")

        else:
            print(f"File {self.pth_path} does not exist. Creating dataset...")
            all_dataset, all_labels, all_file_names = self.get_dataset()
            
            self.data_list = []
            self.split_index_list = []

            self.filename_list = []

            if self.args.use_h5 == False:

                for i in tqdm.tqdm(range(len(all_dataset))):

                    data = all_dataset[i][0]
                    split_index = all_dataset[i][1]

                    if all_labels is not None:
                        label = all_labels[i]
                    filename = all_file_names[i]

                    self.data_list.append(data)
                    self.split_index_list.append(split_index)
                    self.filename_list.append(filename)
                    self.label_list.append(label)

                    data_dict = {
                        "data": self.data_list,
                        "label": self.label_list,
                        "file_name": self.filename_list,
                        "split_idx": self.split_index_list,
                    }

                print(f"Dataset {flag} created!")
                torch.save(data_dict, self.pth_path)
                print(f"File {self.pth_path} saved!")

                self.class_nums = np.zeros(args.num_classes)
                if self.label_list is not None:
                    for label in self.label_list:
                        self.class_nums[label] += 1

                    self.class_ratio = self.class_nums / sum(self.class_nums) * 100

                    print("[Preprocessed {} data information]".format(flag))
                    print("total number of audio data : {}".format(len(self.data_list)))
                    for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                        print("Class {}: {:<4} ({:.1f}%)".format(i, int(n), p))
            else:
                self.class_nums = np.zeros(args.num_classes)

                chunk_size = 1000  # 한 번에 처리할 데이터 개수
                total_chunks = (len(all_dataset) + chunk_size - 1) // chunk_size
                
                with h5py.File(self.pth_path, "w") as file:
                    for chunk in range(total_chunks):
                        start = chunk * chunk_size
                        end = min((chunk + 1) * chunk_size, len(all_dataset))

                        for i in tqdm.tqdm(
                            range(start, end),
                            desc=f"Processing chunk {chunk+1}/{total_chunks}",
                        ):
                            start = chunk * chunk_size
                            end = min((chunk + 1) * chunk_size, len(all_dataset))

                            try:

                                data = all_dataset[i][0]
                                split_index = all_dataset[i][1]
                                filename = all_file_names[i]
                                
                                group = file.create_group(str(i))
                                group.create_dataset(
                                    "audio",
                                    data=data,
                                    compression="gzip",
                                )
                                group.create_dataset(
                                    "split_index", data=split_index, dtype="int32"
                                )
                                group.create_dataset(
                                    "filename", data=filename.encode("utf-8")
                                )
                                
                                label = all_labels[i]
                                label_np = label.numpy()
                                group.create_dataset("label", data=label_np, dtype="int32")
                                self.class_nums[label.item()] += 1
                                    # group.create_dataset("label", data=label.numpy(), dtype="int32")

                                file.flush()  # 각 청크 처리 후 디스크에 쓰기

                            except Exception as e:
                                print(f"Error processing chunk {chunk}: {e}")
                                exit(100)

                    group = file.create_group("info")
                    group.create_dataset("class_num", data=self.class_nums)
                    group.create_dataset(
                        "file_num", data=len(all_dataset), dtype=np.int32
                    )

                del all_dataset

                with h5py.File(self.pth_path, "r") as hf:
                    self.h5_length = int(hf["info"]["file_num"][()])

                    last_idx = self.h5_length - 1
                    print(f"Keys for last index {last_idx}:")
                    for key in hf[str(last_idx)].keys():
                        print(f"  - {key}")

                    try:
                        audio = hf[str(last_idx) + "/audio"][()]
                        # label = self.hf[str(last_idx) + "/label"][()]
                        print(f"Audio shape: {audio.shape}, Label: {label}")
                    except Exception as e:
                        print(f"Error accessing last item: {e}")


                    self.class_ratio = self.class_nums / sum(self.class_nums) * 100

                    print("[Preprocessed {} data information]".format(flag))
                    print("total number of audio data : {}".format(self.h5_length))
                    for i, (n, p) in enumerate(
                        zip(self.class_nums, self.class_ratio)
                    ):
                        print("Class {}: {:<4} ({:.1f}%)".format(i, int(n), p))

    def get_sample(self, row, filename):

        label = None
   
        onset, offset, label = get_data_info(row, self.args.mode)

        # mmvd 만 고려하는 경우.
        if label == -1:
            print(f"Error label is -1, skipping: {filename}")
            return None, None

        if float(onset) == float(offset):
            print(f"Error onset and offset are same, skipping: {filename}")
            return None, None

        if "patient_list_foldwise" in filename:

            return None, None

        filepath = os.path.join(self.target_dir, f"{filename}.wav")

        # check samplerate
        sr = librosa.get_samplerate(filepath)
        # loading with slicing => torchaudio.load(wavfilepath, frame_offset, num_frames)
        audio, _ = torchaudio.load(filepath, int(onset * sr), (int(offset * sr) - int(onset * sr)))

        # error
        if audio.shape[1] == 0:
            print(
                f"2. filename: {filename}, audio: {audio}, audio shape: {audio.shape} label: {label} onset: {onset} offset: {offset}"
            )
            return None, None

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.args.samplerate:
            resample = T.Resample(sr, self.args.samplerate)
            audio = resample(audio)

        if self.args.use_filter is True:

            if "heart" in self.args.mode:
                lowpass_cutoff = self.args.fmax
                highpass_cutoff = self.args.fmin

            if "lung" in self.args.mode:
                lowpass_cutoff = self.args.fmax
                highpass_cutoff = self.args.fmin

            shelf_gain = 4.5
            Q = 0.707

            bq_lp = Biquad(Biquad.LOWPASS, lowpass_cutoff, self.args.samplerate, Q, 0)
            bq_hp = Biquad(Biquad.HIGHPASS, highpass_cutoff, self.args.samplerate, Q, 0)
            bq_ls = Biquad(
                Biquad.LOWSHELF, lowpass_cutoff, self.args.samplerate, Q, shelf_gain
            )
            bq_hs = Biquad(
                Biquad.HIGHSHELF, highpass_cutoff, self.args.samplerate, Q, shelf_gain
            )

            audio = biquad_filter(audio, bq_lp, bq_hp, bq_ls, bq_hs)

        return audio, label

    def get_dataset(self):

        dataset = []
        labels = []
        filenames = []

        for index, filename in tqdm.tqdm(enumerate(self.filenames)):
            # filename exmaple : 200900242_202209010943_M_h_wp100
            annotation = get_annotation(self.target_dir, filename, self.args.mode)
            len_annotation = len(annotation)


            for i in range(len_annotation):
                row = annotation.loc[i]
                audio, label = self.get_sample(row, filename)
                

                if audio is None:
                    continue

                audio = audio.detach().cpu()
                # Split and Pad

                if audio.shape[-1] > self.targetsample:
                    cut_time = audio.shape[-1] // self.targetsample

                    for idx in range(cut_time):
                        cut_audio = audio[
                            ..., self.targetsample * idx : self.targetsample * (idx + 1)
                        ]
                        # cut_audio = torch.Tensor(cut_audio).unsqueeze(dim=0)

                        dataset.append((cut_audio, idx))

                        if label is not None:
                            labels.append(label)

                        filenames.append(filename)

                    if audio.shape[-1] / cut_time > self.targetsample:
                        last_audio = audio[..., self.targetsample * cut_time :]
                        if self.args.pad == "zero":
                            tmp = torch.zeros(1, self.targetsample, dtype=torch.float32)
                            diff = self.targetsample - last_audio.shape[-1]
                            tmp[..., diff // 2 : last_audio.shape[-1] + diff // 2] = (
                                last_audio
                            )
                            pad_data = tmp

                        elif self.args.pad == "repeat":
                            ratio = math.ceil(self.targetsample / last_audio.shape[-1])
                            pad_data = last_audio.repeat(1, ratio)
                            pad_data = pad_data[..., : self.targetsample]

                        dataset.append((pad_data, idx + 1))
                        if label is not None:
                            labels.append(label)
                        filenames.append(filename)

                # Just Pad
                else:
                    if self.args.pad == "repeat":

                        ratio = math.ceil(self.targetsample / audio.shape[-1])
                        audio = audio.repeat(1, ratio)  # dim=(1,ratio)
                        audio = audio[..., : self.targetsample]

                    elif self.args.pad == "zero":
                        tmp = torch.zeros(1, self.targetsample, dtype=torch.float32)
                        diff = self.targetsample - audio.shape[-1]
                        tmp[..., diff // 2 : audio.shape[-1] + diff // 2] = audio

                        audio = tmp

                    dataset.append((audio, 0))
                    if labels is not None:
                        labels.append(label)
                    filenames.append(filename)

        return dataset, torch.tensor(labels), filenames

    def __len__(self):

        if self.args.use_h5 == True:
            return self.h5_length

        return len(self.data_list)

    def __getitem__(self, idx):

        if self.args.use_h5 == False:

            return (
                self.data_list[idx],
                self.label_list[idx],
                self.filename_list[idx],
                self.split_index_list[idx],
            )

        else:
            with h5py.File(self.pth_path, "r") as hf:
                self.h5_length = int(hf["info"]["file_num"][()])

                group = hf[str(idx)]
                audio = group["audio"][()]
                split_index = group["split_index"][()]
                filename = group["filename"][()].decode("utf-8")

                label = None
                
                if "label" in group:
                    # Fetch label
                    label_dataset = group["label"]
                    
                    if label_dataset.shape == () and label_dataset.size > 0:  # Scalar dataset
                        label = label_dataset[()].item()  # Extract the scalar value

                    # if label_dataset.shape and label_dataset.size > 0:  # Ensure it's not empty
                    #     label = label_dataset[()]
                    #     print(f"Read label for index {idx}: {label}")

                    #     # Convert to PyTorch tensor
                        label = torch.tensor(label, dtype=torch.int64)
                    else:
                        print("Label is empty 1. Assigning default value.")
                        print(split_index, filename)
                
                    # Return the required outputs
                    return (
                        audio,
                        label,
                        filename,
                        split_index,
                    )
                
    def check_h5_structure(self):
        with h5py.File(self.pth_path, "r") as hf:
            print("File structure:")
            hf.visit(lambda name: print(name))

            print("\nChecking data:")
            for key in hf.keys():
                if key != "info":
                    group = hf[key]
                    print(f"Group {key}:")
                    for dataset in group.keys():
                        print(
                            f"  - {dataset}: shape {group[dataset].shape}, dtype {group[dataset].dtype}"
                        )
