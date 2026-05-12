import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect4NNet(nn.Module):
    def __init__(self, game, args):
        super().__init__()

        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        c = args.num_channels

        self.conv1 = nn.Conv2d(1, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)

        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)

        self.conv3 = nn.Conv2d(c, c, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(c)

        flat_size = c * self.board_x * self.board_y

        self.fc1 = nn.Linear(flat_size, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.pi = nn.Linear(512, self.action_size)
        self.v = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 1, self.board_x, self.board_y)

        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))

        s = s.view(s.size(0), -1)

        s = F.relu(self.fc_bn1(self.fc1(s)))
        s = F.relu(self.fc_bn2(self.fc2(s)))

        pi = F.log_softmax(self.pi(s), dim=1)
        v = torch.tanh(self.v(s))

        return pi, v