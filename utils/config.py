import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of GAN models.")

    parser.add_argument('--model', type=str, default='DCGAN', choices=['GAN', 'DCGAN', 'WGAN-CP', 'WGAN-GP'])
    parser.add_argument('--is_train', type=str, default='True')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar', 'stl10'],
                            help='The name of dataset')
    parser.add_argument('--download', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--cuda',  type=str, default='False', help='Availability of cuda')

    parser.add_argument('--load_D', type=str, default='False', help='Path for loading Discriminator network')
    parser.add_argument('--load_G', type=str, default='False', help='Path for loading Generator network')
    parser.add_argument('--generator_iters', type=int, default=10000, help='The number of iterations for generator in WGAN model.')
    return check_args(parser.parse_args())


# Checking arguments
def check_args(args):
    # --epoch
    try:
        assert args.epochs >= 1
    except:
        print('Number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larger than or equal to one')

    if args.dataset == 'cifar' or args.dataset == 'stl10':
        args.channels = 3
    else:
        args.channels = 1
    args.cuda = True if args.cuda == 'True' else False
    return args
