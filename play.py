import torch
from models import *
from game import GAME
#from llm import llm, TEXT
from multiprocessing import Pool
from tqdm import tqdm
from dspGuru import HeartsAIPlayer as H
#import os

'''
no_llm = True
if not no_llm:
    if 'y' in input('Local llm is very slow. Proceed? y/N').lower():
        qwen_size = '0.6' # or '8' or '32'
        qwen = llmPlayer(llm(os.path.join(
            os.environ['USERPROFILE'], f'.cache/modelscope/hub/models/Qwen/Qwen3-{qwen_size}B')))
        qwen.ai(TEXT)
'''

GAME.lr = None

def autoplay(players, times=10000000, mp=10000):
    GAME.players = players
    with torch.no_grad():
        if mp:
            with Pool(20) as pool:
                return sum(sum(pool.map(GAME, range(mp))) for i in tqdm(range(times // mp))) / times
        return sum(GAME(i) for i in tqdm(range(times))) / times

def main():
    PLAYER = DeepHearts()
    PLAYER.model.load_state_dict(torch.load('hearts_ai.pth'))
    PLAYER.model.eval()
    for j in [H(0), RANDOM_PLAYER, PLAYER]:
        for k in [[H(1), H(2), H(3)], [RANDOM_PLAYER] * 3, [PLAYER] * 3]:
            print(j.__class__.__name__, '1v3', k[0].__class__.__name__, autoplay([j] + k))
            k[1] = H(2) if j.__class__.__name__ == 'HeartsAI' else j
            print(j.__class__.__name__, '2v2', k[0].__class__.__name__, autoplay([j] + k))

if __name__ == '__main__':
    main()