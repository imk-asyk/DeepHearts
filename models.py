import random
import torch
from torch import nn
from game import getavailable

class MinMaxNorm(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min = min
        self.max = max
    
    def forward(self, x):
        m = x.min()
        return (x - m) / (x.max() - m) * (
            self.max - self.min) + self.min

class DeepHearts:
    def __init__(self):
        layers = []
        input_size = 46
        # stats + now + cardcount + have spade Q, K, A + pass
        for i in [512, 256]:
            layers.append(nn.Linear(input_size, i))
            layers.append(nn.ReLU())
            input_size = i
        layers.append(nn.Linear(input_size, 52))
        layers.append(MinMaxNorm(-4, 4))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers).share_memory()

    def passcards(self, cards, dir, val=False):
        r = torch.where(
            torch.tensor([i % 13 in cards[i // 13] for i in range(52)]),
            self.model(torch.tensor(
                [0] * 38 + [len(i) for i in cards] +
                [sum(i in cards[0] for i in [10, 11, 12])] +
                [i == dir for i in range(3)],
                dtype=torch.float32)), torch.zeros(52))
        result = []
        submit = 0
        for j in torch.multinomial(r, 3):
            i = j.item()
            result.append((i // 13, i % 13))
            if val:
                submit = submit + torch.log(r[j] / r.sum())
        if val:
            return result, submit
        return result
        
    def play(self, played, status, now, cards, val=False):
        available = getavailable(played, now, cards)
        n = sum(([i == k for k in range(4)] + [j]
                 for i, j in now), start=[])
        n += [0] * (15 - len(n))
        r = torch.where(
            torch.tensor([i % 13 in available[i // 13] for i in range(52)]),
            self.model(torch.tensor(
                [sum(played[i * 13: i * 13 + 13]) for i in range(4)]
                + status + played[10: 13] + n + [len(i) for i in cards]
                + [sum(i in cards[0] for i in [10, 11, 12])] + [0] * 3,
                dtype=torch.float32)), torch.zeros(52))
        i = torch.multinomial(r, 1)
        a = i.item()
        if val:
            return (a // 13, a % 13), torch.log(r[i] / r.sum())
        return a // 13, a % 13
    
    def getcards(self, *args):
        pass

    def getpass(self, *args):
        pass

    def gettrick(self, *args):
        pass

class RandomModel:
    def passcards(self, cards, pos, val=False):
        cardlist = [(i, j) for i in range(4) for j in cards[i]]
        random.shuffle(cardlist)
        return cardlist[:3]
        
    def play(self, played, stat, now, cards, val=False):
        available = getavailable(sum(played[13: 26]), now, cards)
        cardlist = [(i, j) for i in range(4) for j in available[i]]
        return random.choice(cardlist)
    
    def getcards(self, *args):
        pass

    def getpass(self, *args):
        pass

    def gettrick(self, *args):
        pass

RANDOM_PLAYER = RandomModel()
'''
# introduce LLMs
class llmPlayer:
    def __init__(self, ai):
        self.ai = ai
    
    def passcards(self, cards, pos, val=False):
        # no assertion since exception will be thrown by the game anyway
        return eval(self.ai(f'PASS {pos}'))
    
    def play(self, played, status, now, cards, val=False):
        ans = eval(self.ai(f'PLAY {now}'))
        assert ans[1] in getavailable(played, now, cards)[ans[0]]
        return ans

    def getcards(self, cards):
        self.ai(f'START {[(i, j) for i in range(4) for j in cards[i]]}')

    def getpass(self, cards):
        self.ai(f'GET {cards}')
    
    def gettrick(self, cards):
        self.ai(f'TRICK {cards}')
'''