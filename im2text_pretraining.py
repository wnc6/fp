import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as torchdata
from tensorboard_logger import configure, log_value
import tqdm
import utils

def parse_arguments():
    """
    Parse command-line arguments for the training script.

    Returns:
        args: argparse.Namespace, parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Wikipedia_Pretraining')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--data_dir', default='data/', help='data directory')
    parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
    parser.add_argument('--train_csv', default='cv/tmp/', help='train csv directory')
    parser.add_argument('--val_csv', default='cv/tmp/', help='validation csv directory')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
    parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
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

def configure_tensorboard(args):
    """
    Configure TensorBoard logger.

    Args:
        args: argparse.Namespace, parsed command-line arguments.
    """
    configure(args.cv_dir + '/log', flush_secs=5)

def train(epoch, counter):
    """
    Train the model for one epoch.

    Args:
        epoch: int, current epoch number.
        counter: int, counter for logging.
    """
    rnet.train()
    losses = []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = Variable(inputs)
        targets = Variable(targets)
        preds = rnet.forward(inputs)
        loss = criterion(preds, targets, torch.ones((inputs.size(0))).cuda()
        if batch_idx % 50 == 0:
            log_value('train_loss_iteration', loss, counter)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu())
        counter += 1
    loss = torch.stack(losses).mean()
    log_value('train_loss_epoch', loss, epoch)
    print(f'E: {epoch} | L: {loss:.3f}')

def test(epoch):
    """
    Test the model on the validation set.

    Args:
        epoch: int, current epoch number.
    """
    rnet.eval()
    losses = []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        preds = rnet.forward(inputs)
        loss = criterion(preds, targets, torch.ones((inputs.size(0))).cuda()
        losses.append(loss.cpu())
    loss = torch.stack(losses).mean()
    log_value('test_loss', loss, epoch)
    print(f'TS: {epoch} | L: {loss:.3f}')
    # Save model
    rnet_state_dict = rnet.module.state_dict() if args.parallel else rnet.state_dict()
    torch.save({'state_dict': rnet_state_dict, 'epoch': epoch}, args.cv_dir + f'/ckpt_E_{epoch}')

def main():
    """
    Main function to run the training and testing loop.
    """
    args = parse_arguments()
    ensure_checkpoint_directory(args)
    utils.save_args(__file__, args)
    rnet = utils.get_model()
    if args.parallel:
        rnet = nn.DataParallel(rnet)
    rnet.cuda()
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(rnet.parameters(), lr=args.lr)
    configure_tensorboard(args)
    trainset, testset = utils.get_dataset(args.train_csv, args.val_csv, pretrain=True)
    trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    start_epoch = 0
    counter = 0
    for epoch in range(start_epoch, start_epoch + args.max_epochs + 1):
        train(epoch, counter)
        if epoch % 1 == 0:
            test(epoch)
        # Assuming lr_scheduler is defined elsewhere, e.g., optim.lr_scheduler.StepLR(optimizer, step_size=args.epoch_step)
        lr_scheduler.step()

if __name__ == "__main__":
    main()
