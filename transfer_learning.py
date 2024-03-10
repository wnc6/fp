import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value
import utils

cudnn.benchmark = True

def parse_arguments():
    """
    Parse command-line arguments for the training script.

    Returns:
        args: argparse.Namespace, parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--data_dir', default='data/', help='data directory')
    parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
    parser.add_argument('--train_csv', default='cv/tmp/', help='train csv directory')
    parser.add_argument('--val_csv', default='cv/tmp/', help='validation csv directory')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
    parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
    parser.add_argument('--load', help='Checkpoints for the pretrained model')
    parser.add_argument('--parallel', action='store_true', default=False, help='use multiple GPUs for training')
    return parser.parse_args()

def ensure_checkpoint_directory(args):
    """
    Ensure the checkpoint directory exists, creating it if necessary.

    Args:
        args: argparse.Namespace, parsed command-line arguments.
    """
    if not os.path.exists(args.cv_dir):
        os.makedirs(args.cv_dir)

def train(epoch, net, trainloader, optimizer):
    """
    Train the model for one epoch.

    Args:
        epoch: int, current epoch number.
        net: torch.nn.Module, the model to train.
        trainloader: DataLoader, the training data loader.
        optimizer: torch.optim.Optimizer, the optimizer.
    """
    net.train()
    matches, losses = [], []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        preds = net.forward(inputs)
        _, pred_idx = preds.max(1)
        match = (pred_idx == targets).float()
        loss = F.cross_entropy(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        matches.append(match.cpu())
        losses.append(loss.cpu())
    accuracy = torch.cat(matches, 0).mean()
    loss = torch.stack(losses).mean()
    log_str = f'E: {epoch} | A: {accuracy:.3f} | L: {loss:.3f}'
    print(log_str)
    log_value('train_accuracy', accuracy, epoch)
    log_value('train_loss', loss, epoch)

def test(epoch, net, testloader):
    """
    Test the model on the validation set.

    Args:
        epoch: int, current epoch number.
        net: torch.nn.Module, the model to test.
        testloader: DataLoader, the validation data loader.
    """
    net.eval()
    matches = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            preds = net.forward(inputs)
            _, pred_idx = preds.max(1)
            match = (pred_idx == targets).float()
            matches.append(match.cpu())
    accuracy = torch.cat(matches, 0).mean()
    log_str = f'TS: {epoch} | A: {accuracy:.3f}'
    print(log_str)
    log_value('test_accuracy', accuracy, epoch)
    # Save model
    net_state_dict = net.module.state_dict() if args.parallel else net.state_dict()
    torch.save({'state_dict': net_state_dict, 'epoch': epoch, 'acc': accuracy}, f'{args.cv_dir}/ckpt_E_{epoch}_A_{accuracy:.3f}')

def main():
    """
    Main function to run the training and testing loop.
    """
    args = parse_arguments()
    ensure_checkpoint_directory(args)
    utils.save_args(__file__, args)
    trainset, testset = utils.get_dataset(args.train_csv, args.val_csv)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    net = utils.get_model()
    if args.parallel:
        net = nn.DataParallel(net)
    net.cuda()
    if args.load:
        checkpoint = torch.load(args.load)
        net.load_state_dict(checkpoint['state_dict'])
    start_epoch = 0
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    configure(args.cv_dir+'/log', flush_secs=5)
    for epoch in range(start_epoch, start_epoch + args.max_epochs + 1):
        train(epoch, net, trainloader, optimizer)
        if epoch % 1 == 0:
            test(epoch, net, testloader)

if __name__ == "__main__":
    main()
