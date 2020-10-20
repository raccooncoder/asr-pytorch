import torch
import torch.nn.functional as F
from utils import GreedyDecoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualise(model, dataset, idx2char, idx):
    model.eval() 
    
    images, labels, target_len, padded_len = dataset[idx]
    
    images, labels = images.to(device), labels.to(device)

    outputs = model(images.unsqueeze(0))
    outputs = F.log_softmax(outputs, dim=2)

    decodes, targets = GreedyDecoder(outputs.transpose(0, 1), labels.unsqueeze(0), target_len.unsqueeze(0), idx2char, replace=False)
    
    return decodes, targets