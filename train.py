import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sampler import InfiniteSamplerWrapper
from models.StyTR import StyTrans, decoder

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.paths = list(Path(root).rglob('*.*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return len(self.paths)

def adjust_learning_rate(optimizer, iteration_count, args):
    lr = args.lr / (1.0 + args.lr_decay * max(0, iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, iteration_count, args):
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    with torch.no_grad():
        network = StyTrans(decoder, args)

    network.train().to(device)

    content_dataset = FlatFolderDataset(args.content_dir, train_transform())
    style_dataset = FlatFolderDataset(args.style_dir, train_transform())
    
    content_iter = iter(data.DataLoader(
        content_dataset, args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))

    style_iter = iter(data.DataLoader(
        style_dataset, args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    optimizer = torch.optim.Adam([
        {'params': network.tiny_vit_encoder.parameters(), 'lr': args.lr * 0.1},
        {'params': network.decode.parameters(), 'lr': args.lr},
        {'params': network.linear_proj.parameters(), 'lr': args.lr}
    ], lr=args.lr)

    for i in tqdm(range(args.max_iter)):
        if i < 1e4:
            warmup_learning_rate(optimizer, i, args)
        else:
            adjust_learning_rate(optimizer, i, args)

        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)
        
        output, loss_c, loss_s, loss_id1, loss_id2 = network(content_images, style_images)

        total_loss = args.content_weight*loss_c \
                        + args.style_weight*loss_s \
                        + (loss_id1 * 70.0) \
                        + (loss_id2 * 1.0)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            output_folder = os.path.join(args.save_dir, "test")
            os.makedirs(output_folder, exist_ok=True)
            output_resized = F.interpolate(output, size=content_images.shape[2:], mode='bilinear', align_corners=False)
            
            out_images = torch.cat((content_images, style_images, output_resized), dim=0)
            save_image(out_images, os.path.join(output_folder, f"{i+1}.jpg"))

        writer.add_scalar('loss_content', loss_c.item(), i+1)
        writer.add_scalar('loss_style', loss_s.item(), i+1)
        writer.add_scalar('loss_identity1', loss_id1.item(), i+1)
        writer.add_scalar('loss_identity2', loss_id2.item(), i+1)
        writer.add_scalar('loss_total', total_loss.item(), i+1)

        if (i+1) % args.save_model_interval == 0 or (i+1)==args.max_iter:
            checkpoint_path = os.path.join(args.save_dir, f"iter_{i+1}.pth")
            torch.save(network.state_dict(), checkpoint_path)

    writer.close()
    print("Training Done Explicitly Clearly.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', default='datasets/train/realworld', type=str)
    parser.add_argument('--style_dir', default='datasets/train/cartoon', type=str)
    parser.add_argument('--save_dir', default='./experiments', type=str)
    parser.add_argument('--log_dir', default='./logs', type=str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--lr_decay', default=1e-5, type=float)
    parser.add_argument('--max_iter', default=160000, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--style_weight', default=10.0, type=float)
    parser.add_argument('--content_weight', default=7.0, type=float)
    parser.add_argument('--n_threads', default=0, type=int)
    parser.add_argument('--save_model_interval', default=10000, type=int)
    args = parser.parse_args()

    main(args)

# python train.py --content_dir "D:/ghibli_data/real" --style_dir "D:/ghibli_data/ghibli"  --batch_size 8  --max_iter 20 --n_threads 0