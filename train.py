"""
    Training procedure for NICE
"""

import argparse
import torch
import torchvision
from torchvision import transforms
from collections import defaultdict
from tqdm import trange
import matplotlib.pyplot as plt
import nice
import pickle

def train(flow, trainloader, optimizer, device):
    flow.train()  # set to training mode
    loss, num_batch = 0, 0
    for inputs in trainloader:
        num_batch += 1
        features, labels = inputs
        features = features.view(features.shape[0], features.shape[1] * features.shape[2] * features.shape[3])
        features = features.to(device)
        optimizer.zero_grad()
        batch_loss = -flow(features).mean()
        loss += batch_loss
        batch_loss.backward()
        optimizer.step()
    return loss / num_batch

def test(flow, testloader, filename, epoch, sample_shape, device):
    flow.eval()  # set to inference mode
    with torch.no_grad():
        samples = flow.sample(100).to(device)
        a,b = samples.min(), samples.max()
        samples = (samples-a)/(b-a+1e-10) 
        samples = samples.view(-1,sample_shape[0],sample_shape[1],sample_shape[2])
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + '/gaussian/' + filename + '_epoch%d.png' % epoch)
        loss, num_batch = 0, 0
        for inputs in testloader:
            num_batch += 1
            features, labels = inputs
            features = features.view(features.shape[0], features.shape[1] * features.shape[2] * features.shape[3])
            batch_loss = -flow(features).mean()
            loss += float(batch_loss)
    return loss / num_batch

def dequantization(x):
    return x + torch.zeros_like(x).uniform_(0., 1./256.)

def main(args):
    #  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu" # for mac
   
    sample_shape = [1,28,28]
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(dequantization) #dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion_mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    model_save_filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'coupling%d_' % args.coupling \
             + 'coupling_type%s_' % args.coupling_type \
             + 'full%d_' % args.full_dim \
             + 'mid%d_' % args.mid_dim \
             + 'hidden%d_' % args.hidden \
             + '.pt'

    flow = nice.NICE(
                prior=args.prior,
                coupling=args.coupling,
                coupling_type=args.coupling_type,
                in_out_dim=args.full_dim, 
                mid_dim=args.mid_dim,
                hidden=args.hidden,
                device=device).to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)

    train_losses = []
    test_losses = []
    filename = f"{args.dataset}_{args.coupling_type}"
    
    for epoch in trange(args.epochs, desc='Training Epochs'):
        train_loss = train(flow, trainloader, optimizer, device)
        train_losses.append(train_loss)
        
        test_loss = test(flow, testloader, filename, epoch+1, sample_shape, device)
        test_losses.append(test_loss)
        
        print(f"\nEpoch {epoch + 1} finished:\n    train loss: {train_loss}, test loss: {test_loss}")
        
        if epoch % 10 == 0:
            torch.save(flow.state_dict(), "./models/" + model_save_filename)

    with open('./losses/gaussian/' + filename + '_train_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)

    with open('./losses/gaussian/' + filename + '_test_losses.pkl', 'wb') as f:
        pickle.dump(test_losses, f)

    with torch.no_grad():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label="Train Loss", color="yellow", linewidth=2)
        ax.plot(test_losses, label="Test Loss", color="purple", linewidth=2)
        ax.set_title("Train and Test Log Likelihood Loss", fontsize=16)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(fname="./loss/gaussian/" + f"{args.dataset}_{args.coupling_type}_loss.png", dpi=300)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='gaussian')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling-type',
                        help='.',
                        type=str,
                        default='affine')
    parser.add_argument('--coupling',
                        help='.',
                        # type=int,
                        default=4)
    parser.add_argument('--full-dim',
                        help='.',
                        type=int,
                        default=784)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
