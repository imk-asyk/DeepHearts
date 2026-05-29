from multiprocessing import Pool
import torch
from game import GAME
from models import *
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

def train(model, lr=0.001, train=10000, batch=100, val=10000, epoch=100):
    with Pool(20) as pool:
        opt = Adam(model.model.parameters(), lr)
        sch = CosineAnnealingLR(opt, epoch)
        curve = []
        for _ in tqdm(range(epoch)):
            model.model.train()
            GAME.lr = lr
            GAME.players = [model] * 4
            for i in range(train // batch):
                opt.zero_grad()
                sum(GAME(i) for i in range(batch)).backward()
                opt.step()
            sch.step()
            GAME.lr = None
            model.model.eval()
            GAME.players = [model] + [RANDOM_PLAYER] * 3
            with torch.no_grad():
                curve.append(sum(i[0] for i in pool.map(GAME, range(val))) / val)
    return curve

def main():
    h = DeepHearts()
    plt.plot(train(h))
    plt.show()
    torch.save(h.model.state_dict(), f'hearts_ai.pth')

if __name__ == '__main__':
    main()