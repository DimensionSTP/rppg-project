import torch
from torch import nn
import torchvision.models as models

'''
Backbone CNN for RhythmNet model is a RestNet-18
'''


class RhythmNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.fc_regression = nn.Linear(1000, 1)
        self.gru_fc_out = nn.Linear(2000, 1)
        self.rnn = nn.GRU(input_size=1000, hidden_size=1000, num_layers=2, batch_first=True, bidirectional=True)
    
    def forward(self, st_maps):
        batched_output_per_clip = []
        gru_input_per_clip = []
        hr_per_clip = []
        
        for t in range(st_maps.size(1)):
            x = self.resnet18(st_maps[:, t, :, :, :]) # (batch=4, 10, 3, 300, 25)
            # Save CNN features per clip for the GRU
            gru_input_per_clip.append(x) # append (batch=4, 1000) > (t=10, batch=4, 1000)
            # Final regression layer for CNN features -> HR (per clip)
            x = self.fc_regression(x) # (batch=4, 1)
            # normalize HR by frame-rate: 30.0 for my dataset
            x = x * 30.0
            batched_output_per_clip.append(x) # append (batch=4, 1) > (t=10, batch=4, 1)
            # input should be (seq_len, batch, input_size)
        # the features extracted from the backbone CNN are fed to a one-layer GRU structure.
        regression_output = torch.stack(batched_output_per_clip, dim=0).permute(1, 2, 0) # (10, batch=4, 1) > (batch=4, 1, 10)
        # Trying out GRU in addition to the regression now.
        gru_input = torch.stack(gru_input_per_clip, dim=0).permute(1, 0, 2) # (10, batch=4, 1000) > (batch=4, 10, 1000)
        gru_output, _ = self.rnn(gru_input) # (batch=4, 10, 1000) > (batch=4, 10, 1000), (batch=4, 1, 1000)
        for i in range(gru_output.size(1)):
            hr = self.gru_fc_out(gru_output[:, i, :]) # (batch=4, 1000) > (batch=4, 1)
            hr_per_clip.append(hr.flatten()) # append (batch=4) > (10, batch=4)
        gru_output_seq = torch.stack(hr_per_clip, dim=0).permute(1, 0) # (10, batch=4) > (batch=4, 10)
        return regression_output, gru_output_seq # (batch=4, 1, 10), (batch=4, 10)


if __name__ == '__main__':
    model = RhythmNet()
    img = torch.rand(4, 10, 3, 300, 25)*255
    reg_out, gru_out = model(img)
    reg_out = reg_out.detach().numpy()
    gru_out = gru_out.detach().numpy()
    print("reg_out: ", reg_out.shape)
    print("gru_out: ", gru_out.shape)
    print("successfully finished")