#conding:utf8
from data.dataset import Mnist
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from torchnet import meter
import argparse
import models
import torch as t
from utils.image_list_id import ImageLabelTXT
from utils.image_list_id import traversPath



def train(args):

    #step1: configure model
    model = getattr(models, args.model)()
    if args.load_model_path:
        model.load(args.load_model_path)
    if args.use_gpu:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        model = model.to(device)

    #step2: data
    train_data = Mnist(args.train_txt_root, train=True)
    val_data = Mnist(args.train_txt_root, train=False)
    train_dataloader = DataLoader(train_data, args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=args.train_batch_size, shuffle=False, num_workers=args.num_workers)

    #step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = args.lr
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    #step4: meters
    loss_meter = meter.AverageValueMeter()  #可计算均值和方差
    confusion_matrix = meter.ConfusionMeter(10)  #混淆矩阵，可计算准确率 K=10 代表一共10类
    previous_loss = 1e100

    #train
    for epoch in range(args.max_epochs):
        loss_meter.reset()
        confusion_matrix.reset()

        train_loss = 0.
        train_acc = 0.
        i=0

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):
            input = Variable(data)
            target = Variable(label)
            if args.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)

            train_loss += loss.item()
            pred = t.max(score, 1)[1]  #返回最大值的索引
            train_correct = (pred == label).sum()
            train_acc += train_correct.item()
            print('epoch ', epoch, ' batch ', i)
            i += 1
            print('Train Loss: %f, Acc: %f' % (loss.item(), train_correct.item() / float(len(data))))

            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            confusion_matrix.add(score.data, target.data)

            # if ii%opt.print_freq==opt.print_freq-1:
            #     vis.plot('loss', loss_meter.value()[0])

        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
            train_data)), train_acc / (len(train_data))))

        #model.save()   #每个epoch保存一次
        prefix = 'checkpoints/' + model.model_name + '_'
        name = prefix + "epoch_" + str(epoch) + ".pth"
        t.save(model.state_dict(), name)

        # validata

        val_cm, val_accuracy = val(model, val_dataloader, args, criterion, val_data)
        # vis.plot('val_accuracy', val_accuracy)
        # vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
        #     epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
        #     lr=lr))

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * args.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]




def val(model, dataloader, args, criterion, val_data):
    '''
    计算模型在验证集上的准确率
    :param model:
    :param dataloader:
    :return:
    '''

    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    confusion_matrix = meter.ConfusionMeter(10)

    for ii, (data, label) in tqdm(enumerate(dataloader)):
        val_input = Variable(data)
        val_label = Variable(label.type(t.LongTensor))
        if args.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)

        loss = criterion(score, val_label)
        eval_loss += loss.item()
        pred = t.max(score, 1)[1]
        num_correct = (pred == val_label).sum()
        eval_acc += num_correct.item()

        confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            val_data)), eval_acc / (len(val_data))))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def main():
    parser = argparse.ArgumentParser(description='pyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='MnistNet',
                        help='choice model to train (default: MnistNet)')
    parser.add_argument('--train_txt_root', type=str, default='./DataList.txt',
                        help='input txt with train data list (default: ./DataList.txt)')
    parser.add_argument('--test_txt_root', type=str, default='./TestList.txt',
                        help='input txt with test data list (default: ./TestList.txt)')
    parser.add_argument('--train_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--max_epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_decay', type=float, default=0.5, metavar='LR_DECAY',
                        help='learning rate decay (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0e-5, metavar='WEIGHT_DECAY',
                        help='weight decay (default: 0e-5)')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--result_file', type=str, default='result.csv',
                        help='result file')
    parser.add_argument('--load_model_path', type=str, default="checkpoints/mnistNet_epoch_1.pth",
                        help='The path for loading the current Model (default: "")')

    args = parser.parse_args()
    print(args)
    train(args)



if __name__=='__main__':
    root = 'F:/proj/mnist/mnist_train/'
    fileList = []
    traversPath(root, fileList)
    ImageLabelTXT(fileList, "DataList.txt")
    main()

