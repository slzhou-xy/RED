import torch.nn as nn
from Model import Date2VecConvert
import datetime
import torch


class Date2vec(nn.Module):
    def __init__(self):
        super(Date2vec, self).__init__()
        self.d2v = Date2VecConvert(model_path="./d2v_model/d2v_64dim.pth")

    def forward(self, time_seq):
        one_list = []
        for timestamp in time_seq:
            t = datetime.datetime.fromtimestamp(timestamp)
            t = [t.hour, t.minute, t.second, t.year, t.month, t.day]
            x = torch.Tensor(t).float()
            embed = self.d2v(x)
            one_list.append(embed)

        one_list = torch.vstack(one_list).numpy()

        return one_list
