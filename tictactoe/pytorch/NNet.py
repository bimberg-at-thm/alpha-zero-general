import os
import time
import numpy as np
import torch
import torch.optim as optim

import sys
sys.path.append('..')

from utils import *
from tqdm import tqdm
from NeuralNet import NeuralNet
from .TicTacToeNNet import TicTacToeNNet


args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = TicTacToeNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.nnet.to(self.device)

    def train(self, examples):
        """
        examples: list of (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print(f'EPOCH ::: {epoch + 1}')

            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                boards = torch.tensor(np.asarray(boards), dtype=torch.float32).to(self.device)
                target_pis = torch.tensor(np.asarray(pis), dtype=torch.float32).to(self.device)
                target_vs = torch.tensor(np.asarray(vs), dtype=torch.float32).to(self.device)

                out_pi, out_v = self.nnet(boards)

                loss_pi = self.loss_pi(target_pis, out_pi)
                loss_v = self.loss_v(target_vs, out_v)

                total_loss = loss_pi + loss_v

                # record loss
                pi_losses.update(loss_pi.item(), boards.size(0))
                v_losses.update(loss_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with shape board_x x board_y
        returns: pi, v
        """
        start = time.time()

        self.nnet.eval()

        board = torch.tensor(board.astype(np.float32), dtype=torch.float32)
        board = board.unsqueeze(0).to(self.device)

        with torch.no_grad():
            pi, v = self.nnet(board)

        # Wichtig: Netzwerk gibt log_softmax zurück.
        # MCTS erwartet normale Wahrscheinlichkeiten.
        pi = torch.exp(pi).data.cpu().numpy()[0]
        v = v.data.cpu().numpy()[0]

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time() - start))

        return pi, v

    def loss_pi(self, targets, outputs):
        """
        Cross entropy für Zielverteilung aus MCTS.
        targets: Wahrscheinlichkeitsverteilung
        outputs: log_softmax
        """
        return -torch.sum(targets * outputs) / targets.size(0)

    def loss_v(self, targets, outputs):
        """
        Mean squared error für Value Head.
        """
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size(0)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)

        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists!")

        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)

        if not os.path.exists(filepath):
            raise ValueError(f"No model in path '{filepath}'")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.nnet.load_state_dict(checkpoint['state_dict'])