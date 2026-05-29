import bisect
import random
import torch

def getavailable(broken, now, cards):
    available = cards.copy()
    if not now:
        if 0 in cards[2]:
            return [[], [], [0], []]
        if not broken and sum(len(cards[i]) for i in [0, 2, 3]):
            available[1] = []
    elif cards[now[0][0]]:
        available = [cards[i] if i == now[0][0] else [] for i in range(4)]
    elif sum(len(i) for i in cards) == 13 > (10 in cards[0]) + len(cards[1]):
        # first round, having non-score
        available[1] = []
        if 10 in cards[0]:
            available[0] = cards[0].copy()
            available[0].remove(10)
    return available

def pov(stat, i):
    return [stat[(i + j) % 4] for j in range(3)] +\
    [stat[(i + j) % 4 + 4] for j in range(3)] +\
    [stat[(i + j) % 4 + 8] for j in range(3)] +\
    [stat[(i + j) % 4 + 12] for j in range(3)] +\
    [stat[(i + j) % 4 + 16] for j in range(4)]

class Game:
    def __init__(self, player=None, lr=0.0001, c=6.5):
        self.players = player
        self.lr = lr
        self.c = c
    
    def __call__(self, pos):
        # start
        if self.lr:
            tensor = [0] * 4
        pos %= 4
        played = [0] * 52
        stat = [0] * 20
        cards = [(i, j) for i in range(4) for j in range(13)]
        players_cards = [[[], [], [], []], [[], [], [], []],
                        [[], [], [], []], [[], [], [], []]]
        random.shuffle(cards)
        for i in range(4):
            for j, k in cards[i * 13: i * 13 + 13]:
                bisect.insort(players_cards[i][j], k)
            self.players[i].getcards(players_cards[i])
        if 3 - pos:
            passes = [self.players[i].passcards(
                players_cards[i], [1, 3, 2][pos], self.lr) for i in range(4)]
            if self.lr:
                for i in range(4):
                    tensor[i] = tensor[i] + passes[i][1]
                    passes[i] = passes[i][0]
            for i in range(4):
                for j, k in passes[i]:
                    players_cards[i][j].remove(k)
                for j, k in passes[i - [1, 3, 2][pos]]:
                    bisect.insort(players_cards[i][j], k)
                self.players[i].getpass(passes[i - [1, 3, 2][pos]])
        for i in range(4):
            if 0 in players_cards[i][2]:
                first = i
        # step
        for j in range(13):
            now = []
            for i in range(4):
                now.append(self.players[first + i - 4].play
                           (played, pov(stat, i), now,
                            players_cards[first + i - 4], self.lr))
                if self.lr:
                    tensor[first + i - 4] = tensor[first + i - 4] + now[-1][1]
                    now[-1] = now[-1][0]
                players_cards[first + i - 4][now[i][0]].remove(now[i][1])
                played[now[i][0] * 13 + now[i][1]] = 1
                if now[i][0] - now[0][0]:
                    stat[now[0][0] * 4 + (i + first) % 4] = 1
            lose = -1
            for i in range(4):
                self.players[i].gettrick(now)
                if now[i][0] == now[0][0] and lose < now[i][1]:
                    lose = now[i][1]
                    second = i
            first = (first + second) % 4
            for i in now:
                if i[0] == 1:
                    stat[first + 16] += 1
                if i == (0, 10):
                    stat[first + 16] += 13
        # final
        for i in range(4):
            if stat[16 + i] == 26:
                for j in range(4):
                    stat[16 + j] = 26 - stat[16 + j]
                break
        if not self.lr:
            return torch.tensor(stat[16:])
        return sum((stat[16 + i] - self.c) * tensor[i] for i in range(4))

GAME = Game()