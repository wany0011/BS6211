import torch
import torch.nn as nn
import torch.nn.functional as F


class CurveTraceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, repetition_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.repetition_size = repetition_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=5, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(self.hidden_size, 2)
        self.linear2 = nn.Linear(8, output_size)

    def forward(self, x):
        # print("begin")
        # print(x.size())
        x, _ = self.lstm(x)
        # print(x.size())
        # print(x)
        x = x.view(-1, self.repetition_size, 2, self.hidden_size)
        # print(x.size())
        # print(x)
        x = x[:, :, -1, :]
        # print(x)
        # print(x.size())

        # x=torch.relu(self.linear2(torch.relu(self.linear1(x))))
        x = torch.relu(self.linear1(x))
        # print(x.size())
        return x
