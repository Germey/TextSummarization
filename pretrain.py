from word2vec.process import process
from word2vec.train import train
import time

if __name__ == '__main__':
    print('Start Processing...')
    process()
    
    time.sleep(5)
    
    print('Start Training...')
    train()
