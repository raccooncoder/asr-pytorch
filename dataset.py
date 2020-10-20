import wandb
import torch
import torchaudio
from torch.utils.data import Dataset

class LJSpeech(Dataset):
    def __init__(self, X, y, wav_proc, config, train=True):
        super().__init__()
        self.names = X
        self.labels = y
        self.wav_proc = wav_proc
        self.train = train
        self.config = config
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):        
        img = torch.zeros(self.config .melspec_n_mels, self.config .img_pad_len)
        wav, sr = torchaudio.load(self.config .dataset_path + 'wavs/' + self.names[idx] + '.wav')
            
        mel_spectrogram = torch.log(self.wav_proc(wav) + 1e-9).squeeze(0)
        img[:, :mel_spectrogram.size(1)] = mel_spectrogram
        
        target = torch.tensor(self.labels[idx] + [self.config .txt_pad_idx] * (self.config .txt_pad_len - len(self.labels[idx])))
        target_len = torch.tensor(len(self.labels[idx]) + 1)
        padded_len = torch.tensor(self.config .img_pad_len)
        
        return img, target, target_len, padded_len