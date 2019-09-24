import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
import torch
import random
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train semantic seg')
    parser.add_argument('--log', default=None)
    parser.add_argument('--b', default=0, type=int)

    args = parser.parse_args()

    return args


def main(args):

    logs = args.log.split(',')

    plt.figure()
    for i, path in enumerate(logs):
        lines = open(path).readlines()
        losses = []
        iterations = []
        dice = []

        for line in lines:
            if 'Starting epoch' in line and 'train' in line:
                epoch = int(line.split('|')[0].split(':')[-1])

            if 'loss' in line and 'dice' not in line and 'summary' not in line:
                # print(line)
                iterinfo, loss = line.split(',')
                _iter, iters_pre_epoch = iterinfo.split(':')[-1].split('/') 
                loss = float(loss.split(' ')[-1])    
                iterations.append(epoch*int(iters_pre_epoch)+int(_iter))
                losses.append(loss)

        plt.plot(iterations[args.b:], losses[args.b:])
    
    plt.legend(logs)
    plt.show()
    # plt.show()

if __name__ == '__main__':
    main(parse_args())

