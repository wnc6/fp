import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboard_logger import configure, log_value
from torch.utils.data import DataLoader
from data_loader import OpenI
from xray_utils import transform_xray, get_model
import utils

cudnn.benchmark = True

def parse_arguments():
    """
    Parse command-line arguments for the training script.

    Returns:
        args: argparse.Namespace, parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Radiograph_Pretraining')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--cv_dir', default='cv/', help='checkpoint directory (models and logs are saved here)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
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

def train(epoch, counter, rnet, trainloader, criterion, optimizer):
    """
    Train the model for one epoch.

    Args:
        epoch: int, current epoch number.
        counter: int, counter for logging.
        rnet: nn.Module, the model to train.
        trainloader: DataLoader, the training data loader.
        criterion: nn.Module, the loss function.
        optimizer: torch.optim.Optimizer, the optimizer.
    """
    rnet.train()
    losses = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        preds = rnet.forward(inputs)
        loss = criterion(preds, targets, torch.ones((inputs.size(0))).to(device))
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

def test(epoch, rnet, devloader, criterion):
    """
    Test the model on the validation set.

    Args:
        epoch: int, current epoch number.
        rnet: nn.Module, the model to test.
        devloader: DataLoader, the validation data loader.
        criterion: nn.Module, the loss function.
    """
    rnet.eval()
    losses = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(devloader):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = rnet.forward(inputs)
            loss = criterion(preds, targets, torch.ones((inputs.size(0))).to(device))
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
    train_set = OpenI('train', 'data', transform_xray())
    dev_set = OpenI('dev', 'data', transform_xray(), doc2vec_file="report2vec.model")
    test_set = OpenI('test', 'data', transform_xray(), doc2vec_file="report2vec.model")
    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
    devloader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    rnet = get_model(50)
    if args.parallel:
        rnet = nn.DataParallel(rnet)
    rnet.cuda()
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(rnet.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epoch_step, gamma=args.wd)
    configure(args.cv_dir+'/log', flush_secs=5)
    start_epoch = 0
    counter = 0
    for epoch in range(start_epoch, start_epoch + args.max_epochs + 1):
        train(epoch, counter, rnet, trainloader, criterion, optimizer)
        if epoch % 1 == 0:
            torch.cuda.empty_cache()
            test(epoch, rnet, devloader, criterion)
        scheduler.step()

if __name__ == "__main__":
    main()
