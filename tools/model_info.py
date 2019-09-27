import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train semantic seg')
    parser.add_argument('--folder', metavar='N', type=str, nargs='+')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--postfix', default='best', type=str)

    args = parser.parse_args()

    return args

def main(args):

    prefix = '/DATA5_DB8/data/yanjia/results/'
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%(args.gpu)

    for i, path in enumerate(args.folder):
        model_path = prefix+path+'/model_%s.pth'%args.postfix
        model = torch.load(model_path)
        print('----------------')
        print(path)
        print(model['epoch'])
        # print(model['best_score'])
        print(model['best_loss'])

if __name__ == '__main__':
    main(parse_args())