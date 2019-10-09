import torch
import argparse
import os

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
        model_paths = os.listdir(prefix+path)
        print('----------------')
        print(path)
        for model_path in model_paths:
            abs_model_path = prefix+path+'/'+model_path
            model = torch.load(abs_model_path)
            print('*****')
            print(model_path)
            print(model['epoch'])
            print(model['best_score'])
            print(model['best_loss'])

if __name__ == '__main__':
    main(parse_args())