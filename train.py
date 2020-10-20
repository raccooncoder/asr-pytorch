import wandb
import torch 
import torch.nn.functional as F
from utils import GreedyDecoder, wer, cer

config = wandb.config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(epoch, model, optimizer, error, dataloader, log_freq=config.log_freq):
    model.train() #don't forget to switch between train and eval!
    
    running_loss = 0.0 #more accurate representation of current loss than loss.item()

    for i, (images, labels, target_len, padded_len) in enumerate(dataloader):
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
                    epoch, (i+ 1) * len(images), len(dataloader.dataset),
                    100. * (i + 1) / len(dataloader), running_loss / log_freq))
            
            wandb.log({"Loss": running_loss / log_freq})
            
            running_loss = 0.0
            
def evaluate(epoch, model, optimizer, error, dataloader, idx2char):
    model.eval() 
    loss = 0
    correct = 0
    test_cer = []
    test_wer = []
    
    with torch.no_grad():
        for i, (images, labels, target_len, padded_len) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            outputs = F.log_softmax(outputs, dim=2)

            loss += error(outputs, labels, padded_len, target_len).item()
            
            decoded_preds, decoded_targets = GreedyDecoder(outputs.transpose(0, 1), labels, target_len, idx2char)
            
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
    
    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    loss /= len(dataloader)
    
    wandb.log({"Val loss": loss, 
               "WER": avg_wer, 
               "CER": avg_cer})
    
    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(loss, avg_cer, avg_wer))