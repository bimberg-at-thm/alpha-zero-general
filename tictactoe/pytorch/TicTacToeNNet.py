import torch
import torch.nn as nn
import torch.nn.functional as F


class TicTacToeNNet(nn.Module):
    def __init__(self, game, args):
        super().__init__()

        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        c = args.num_channels

        self.conv1 = nn.Conv2d(1, c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)

        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)

        self.conv3 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c)

        self.conv4 = nn.Conv2d(c, c, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm2d(c)

        conv_out_x = self.board_x - 2
        conv_out_y = self.board_y - 2
        flat_size = c * conv_out_x * conv_out_y

        self.fc1 = nn.Linear(flat_size, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)

        self.fc_pi = nn.Linear(512, self.action_size)
        self.fc_v = nn.Linear(512, 1)

    def forward(self, s):
        # input: batch x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)

        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))

        s = s.view(s.size(0), -1)

        s = F.dropout(
            F.relu(self.bn_fc1(self.fc1(s))),
            p=self.args.dropout,
            training=self.training,
        )

        s = F.dropout(
            F.relu(self.bn_fc2(self.fc2(s))),
            p=self.args.dropout,
            training=self.training,
        )

        pi = F.log_softmax(self.fc_pi(s), dim=1)
        v = torch.tanh(self.fc_v(s))

        return pi, v