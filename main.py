import wandb
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import json

with open('config.json', 'r') as f:
    config = json.load(f)

wandb.init(config=config, project="asr-dlaudio")
config = wandb.config

from utils import set_seed
from train import train, evaluate
from dataset import LJSpeech
from visualise import visualise
from model import ASRNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print(device)

set_seed(config.seed)

df = pd.read_csv(config.dataset_path + "metadata.csv", names=['id', 'gt', 'gt_letters_only'], sep="|")
df = df.dropna()

chars = "qwertyuiopasdfghjklzxcvbnm "
char2idx = {}
idx2char = {}

for idx, char in enumerate(chars):
    char2idx[char] = idx
    idx2char[idx] = char

def clean(s):
    whitelist = "qwertyuiopasdfghjklzxcvbnm "
    return ''.join(filter(whitelist.__contains__, s.lower()))

def f_char2idx(s):
    ans = []
    for char in s:
        ans.append(char2idx[char])
        
    return ans 
    
df['gt_clean'] = df['gt_letters_only'].apply(clean)
df['char2idx'] = df['gt_clean'].apply(f_char2idx)

X_train, X_test, y_train, y_test  = train_test_split(list(df['id']), list(df['char2idx']), train_size=.9)

train_wav_proc = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=config.melspec_sample_rate, 
                                                                    n_mels=config.melspec_n_mels, 
                                                                    n_fft=config.melspec_n_fft, 
                                                                    hop_length=config.melspec_hop_length, 
                                                                    f_max=config.melspec_f_max),
                        torchaudio.transforms.FrequencyMasking(freq_mask_param=config.aug_freq_mask_param),
                        torchaudio.transforms.TimeMasking(time_mask_param=config.aug_time_mask_param)
                        )

test_wav_proc = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=config.melspec_sample_rate, 
                                                                    n_mels=config.melspec_n_mels, 
                                                                    n_fft=config.melspec_n_fft, 
                                                                    hop_length=config.melspec_hop_length, 
                                                                    f_max=config.melspec_f_max))

train_dataset = LJSpeech(X_train, y_train, train_wav_proc, config, train=True)
test_dataset = LJSpeech(X_test, y_test, test_wav_proc, config, train=False)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.dataloader_num_workers)
val_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.dataloader_num_workers) 
    
model = ASRNet()
model = model.to(device)

learning_rate = config.learning_rate
error = nn.CTCLoss(blank=config.ctc_blank).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(config.num_epochs): 
    train(epoch, model, optimizer, error, train_loader)
    #lr_scheduler.step()
    if (epoch + 1) % config.eval_freq == 0:
        evaluate(epoch, model, optimizer, error, val_loader, idx2char)

for idx in range(10):
    print('Example {}:'.format(idx))
    print(visualise(model, test_dataset, idx2char, idx)[0][0])
    print('------------------------------------')
    print(visualise(model, test_dataset, idx2char, idx)[1][0] + '\n')

torch.save(model.state_dict(), 'checkpoints/vanilla_lstm_LJSpeech.pth')

wandb.save('checkpoints/vanilla_lstm_LJSpeech.pth')