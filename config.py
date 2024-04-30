import argparse as arg

parser = arg.ArgumentParser()

# data organization parameters
parser.add_argument('--root_path',default=None,help="model output directory")
parser.add_argument('--model_dir',default='checkpoints',help="model output directory")
parser.add_argument('--load_model',default=None,help="optional model file to initialize with")
parser.add_argument("--result_dir", type=str, help="results folder", dest="result_dir", default='./Result')
parser.add_argument('--train_tb',default='img/train')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--n_cas', default=1,help='recursive_cascade_level')
parser.add_argument('--batch_size', type=int, default=4, help='batch size (default: 1)')
parser.add_argument('--epoch',default=1000 , type= int,help='number of training epoch')
parser.add_argument('--img_shape',default=[32,128,128])
parser.add_argument('--save_per_epoch', type=int, default=1, help='frequency of model saves (default: 100)')
parser.add_argument('--tb_save_freq', type=int, default=100, help='frequency of tensorboard (default: 100)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

# test
parser.add_argument('--test_tb',default='img/test')
parser.add_argument('--test_load_model',default='checkpoints/parnet_best-CHAOS-32128128.pt',help="optional model file to initialize with")
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size (default: 1)')

args = parser.parse_args()