import glob
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader

import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm_notebook as tqdm

import math
import random
import pandas as pd
import numpy as np

num_epochs = 50
batch_size = 64
n_cats = 10

import wandb
wandb.init(project="asr-dlaudio")

# reproducibility
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
set_seed(13)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print(device)

import pandas as pd
df = pd.read_csv("LJSpeech-1.1/metadata.csv", names=['id', 'gt', 'gt_letters_only'], sep="|")
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

class LJSpeech(Dataset):
    def __init__(self, X, y, train=True):
        super().__init__()
        self.names = X
        self.labels = y
        self.train = train
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        img_padding_length = 870
        txt_padding_length = 200
        
        img = torch.zeros(1, 64, img_padding_length)
        wav, sr = torchaudio.load('LJSpeech-1.1/wavs/' + self.names[idx] + '.wav')
        
        if self.train:
            wav_proc = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=64, n_fft=1024, hop_length=256, f_max=8000),
                                        torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
                                        torchaudio.transforms.TimeMasking(time_mask_param=100)
                                    )
        else:
            wav_proc = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=64, n_fft=1024, hop_length=256, f_max=8000),
                                    )
            
        mel_spectrogram = torch.log(wav_proc(wav) + 1e-9)
        img[0, :, :mel_spectrogram.size(2)] = mel_spectrogram
        
        
        target = torch.tensor(self.labels[idx] + [28] * (txt_padding_length - len(self.labels[idx])))
        target_len = torch.tensor(len(self.labels[idx]) + 1)
        padded_len = torch.tensor(img_padding_length)
        
        return img.reshape(64, img_padding_length), target, target_len, padded_len

X_train, X_test, y_train, y_test  = train_test_split(list(df['id']), list(df['char2idx']), train_size=.9)

train_dataset = LJSpeech(X_train, y_train, train=True)
test_dataset = LJSpeech(X_test, y_test, train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

class ASRNet(nn.Module):    
    def __init__(self):
        super(ASRNet, self).__init__()
          
        self.lstm = nn.LSTM(input_size=64, hidden_size=512, bidirectional=True, num_layers=2)
        self.clf = nn.Linear(1024, 29)
        
    def forward(self, x):
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.clf(x)
        return x     
    
model = ASRNet()

model = model.to(device)

learning_rate = 0.002

error = nn.CTCLoss(blank=28).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# helper funcs

def int_to_text(labels):
    string = ""
    for i in labels:
        try:
            string += idx2char[i][0]
        except:
            string += ''
        
    return ''.join(string).replace('', ' ')

def GreedyDecoder(output, labels, label_lengths, blank_label=27, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(int_to_text(decode))
    return decodes, targets

import editdistance


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = editdistance.eval(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = editdistance.eval(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        print(reference)
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer

def train(epoch, log_freq=30):
    model.train() #don't forget to switch between train and eval!
    
    running_loss = 0.0 #more accurate representation of current loss than loss.item()

    for i, (images, labels, target_len, padded_len) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        outputs = F.log_softmax(outputs, dim=2)

        loss = error(outputs, labels, padded_len, target_len)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10, norm_type=2)
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % log_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (i+ 1) * len(images), len(train_loader.dataset),
                    100. * (i + 1) / len(train_loader), running_loss / log_freq))
            
            wandb.log({"Loss": running_loss / log_freq})
            
            running_loss = 0.0
            
def evaluate(data_loader):
    model.eval() 
    loss = 0
    correct = 0
    test_cer = []
    test_wer = []
    
    with torch.no_grad():
        for i, (images, labels, target_len, padded_len) in enumerate(tqdm(data_loader)):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            outputs = F.log_softmax(outputs, dim=2)

            loss += error(outputs, labels, padded_len, target_len).item()
            
            decoded_preds, decoded_targets = GreedyDecoder(outputs.transpose(0, 1), labels, target_len)
            
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
    
    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    loss /= len(data_loader)
    
    wandb.log({"Val loss": loss, 
               "WER": avg_wer, 
               "CER": avg_cer})
    
    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(loss, avg_cer, avg_wer))

for epoch in range(num_epochs): 
    train(epoch)
    #lr_scheduler.step()
    if (epoch + 1) % 10 == 0:
        evaluate(val_loader)

def int_to_text(labels):
    string = ""
    for i in labels:
        try:
            string += idx2char[i][0]
        except:
            string += ''
        
    return ''.join(string)

def GreedyDecoder(output, labels, label_lengths, blank_label=27, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(int_to_text(decode))
    return decodes, targets

def visualise(idx):
    model.eval() 
    
    images, labels, target_len, padded_len = test_dataset[idx]
    
    images, labels = images.to(device), labels.to(device)

    outputs = model(images.unsqueeze(0))
    outputs = F.log_softmax(outputs, dim=2)

    decodes, targets = GreedyDecoder(outputs.transpose(0, 1), labels.unsqueeze(0), target_len.unsqueeze(0))
    
    return decodes, targets

for idx in range(10):
    print('Example {}:'.format(idx))
    print(visualise(idx)[0][0])
    print('------------------------------------')
    print(visualise(idx)[1][0] + '\n')

torch.save(model.state_dict(), 'vanilla_lstm_LJSpeech.pth')

wandb.save('vanilla_lstm_LJSpeech.pth')