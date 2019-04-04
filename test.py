#conding:utf8
from data.dataset import Mnist
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import argparse
import models
import torch as t



def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)

def test(args):
    model = getattr(models, args.model)().eval()
    if args.load_model_path:
        model.load(args.load_model_path)
    if args.use_gpu:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        model = model.to(device)

    test_data = Mnist(args.test_txt_root, test=True)
    test_dataloader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input = t.autograd.Variable(data)
        if args.use_gpu:
            input = input.cuda()
        score = model(input)
        #probability = t.nn.functional.softmax(score)[:,0].data.tolist()

        label = score.max(dim=1)[1].data.tolist()

        #batch_results = [(path_,probability_) for path_,probability_ in zip(path,probability) ]
        batch_results = [(path_, label_) for path_, label_ in zip(path, label)]
        results += batch_results

    write_csv(results, args.result_file)

    return results

def main():
    parser = argparse.ArgumentParser(description='pyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='MnistNet',
                        help='choice model to train (default: MnistNet)')
    parser.add_argument('--test_txt_root', type=str, default='./TestList.txt',
                        help='input txt with test data list (default: ./TestList.txt)')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--result_file', type=str, default='result.csv',
                        help='result file')
    parser.add_argument('--load_model_path', type=str, default="checkpoints/mnistNet_epoch_1.pth",
                        help='The path for loading the current Model (default: "")')

    args = parser.parse_args()
    print(args)
    if test(args):
        print("finish!\nresult see {}".format(args.result_file))
    else:
        print("sorry!...")


if __name__ == '__main__':
    main()