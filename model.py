from torch import nn

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