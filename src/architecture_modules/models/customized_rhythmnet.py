import torch
from torch import nn

import timm

"""
Backbone CNN for RhythmNet model is a RestNet-18
"""


class CustomizedRhythmNet(nn.Module):
    def __init__(
        self,
        cnn_model: str,
        cnn_pretrained: bool,
        rnn_type: str,
        rnn_num_layers: int,
        direction: str,
    ):
        super().__init__()

        if cnn_pretrained:
            self.cnn_model = timm.create_model(cnn_model, pretrained=True)
        else:
            self.cnn_model = timm.create_model(cnn_model, pretrained=False)

        if direction == "bi":
            self.fc_rnn = nn.Linear(2000, 1)
            self.bidirectional = True
        elif direction == "uni":
            self.fc_rnn = nn.Linear(1000, 1)
            self.bidirectional = False
        else:
            self.fc_rnn = nn.Linear(2000, 1)
            self.bidirectional = True

        self.fc_regression = nn.Linear(1000, 1)

        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=1000,
                hidden_size=1000,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=1000,
                hidden_size=1000,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
        else:
            self.rnn = nn.GRU(
                input_size=1000,
                hidden_size=1000,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
            )

    def forward(self, st_maps):
        batched_output_per_clip = []
        rnn_input_per_clip = []
        hr_per_clip = []

        for t in range(st_maps.size(1)):
            x = self.cnn_model(st_maps[:, t, :, :, :])
            # Save CNN features per clip for the RNN
            rnn_input_per_clip.append(x)
            # Final regression layer for CNN features -> HR (per clip)
            x = self.fc_regression(x)
            # normalize HR by frame-rate: 30.0 for my dataset
            x = x * 30.0
            batched_output_per_clip.append(x)
            # input should be (seq_len, batch, input_size)
        # the features extracted from the backbone CNN are fed to a one-layer RNN structure.
        regression_output = torch.stack(batched_output_per_clip, dim=0).permute(1, 2, 0)
        # Trying out RNN in addition to the regression now.
        rnn_input = torch.stack(rnn_input_per_clip, dim=0).permute(1, 0, 2)
        rnn_output, _ = self.rnn(rnn_input)
        for i in range(rnn_output.size(1)):
            hr = self.fc_rnn(rnn_output[:, i, :])
            hr_per_clip.append(hr.flatten())
        rnn_output_seq = torch.stack(hr_per_clip, dim=0).permute(1, 0)
        return regression_output, rnn_output_seq


if __name__ == "__main__":
    model = CustomizedRhythmNet("seresnext26d_32x4d", False, "gru", 10, "bi")
    img = torch.rand(4, 10, 3, 300, 25) * 255
    reg_out, rnn_out = model(img)
    reg_out = reg_out.detach().numpy()
    rnn_out = rnn_out.detach().numpy()
    print("reg_out: ", reg_out.shape)
    print("rnn_out: ", rnn_out.shape)
    print("successfully finished")
