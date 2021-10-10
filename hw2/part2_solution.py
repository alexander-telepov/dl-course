# Don't erase the template code, except "Your code here" comments.

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from timeit import default_timer
from torch.utils.tensorboard import SummaryWriter
import os

def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val', the dataloader should be deterministic.
    
    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train' or 'val'
        
    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    class RandomNoise:
        def __init__(self, sigma=0.025):
            self.sigma = sigma

        def __call__(self, x):
            noise = torch.normal(0, self.sigma, size=x.shape)
            return x + noise
    
    preprocessing_train = transforms.Compose([
        transforms.RandomApply([
            # transforms.GaussianBlur(4),
            transforms.RandomResizedCrop(64, scale=(0.66, 1.0), ratio=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(20, (0.2, 0.2)),
        ], p=0.7),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomApply([RandomNoise(),], p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    preprocessing_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    if kind == 'val':
        dataset = torchvision.datasets.ImageFolder(os.path.join(path, kind), transform=preprocessing_val)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    elif kind == 'train':
        dataset = torchvision.datasets.ImageFolder(os.path.join(path, kind), transform=preprocessing_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    else:
        raise ValueError(f'Unknown kind: {kind}')

    return dataloader

def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.
    
    return:
    model:
        `torch.nn.Module`
    """
    class BasicBlock_(nn.Module):
        def __init__(self, in_channels, out_channels, downsample=True, relu=True):
            super().__init__()
            s = 2 if downsample else 1    
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, s, 1, groups=in_channels),
                nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.downsample = downsample
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
            self.relu = relu

        def forward(self, x):
            out = self.backbone(x)
            identity = F.avg_pool2d(x, 2) if self.downsample else x
            out += self.shortcut(identity)
            if self.relu:
                out = F.leaky_relu(out)
            return out

    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, downsample=True, relu=True, r=2):
            super().__init__()
            self.basic_block = BasicBlock_(in_channels, out_channels, downsample, relu)
            self.squeeze_and_excitation_block = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(out_channels, out_channels // r),
                nn.LeakyReLU(inplace=True),
                nn.Linear(out_channels // r, out_channels),
                nn.Sigmoid()
            )
    
        def forward(self, x):
            out = self.basic_block(x)
            out *= self.squeeze_and_excitation_block(out)[..., None, None]
    
            return out
    
    class MobileResnet(nn.Module):
        def __init__(self, num_classes=200):
            super(MobileResnet, self).__init__()
            self.conv = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.bn = nn.BatchNorm2d(64)
            self.backbone = nn.Sequential(
                BasicBlock(64,  64,  True, r=4),
                BasicBlock(64,  128, True, r=4),
                BasicBlock(128, 128, False, r=4),
                BasicBlock(128, 256, True, r=8),
                BasicBlock(256, 256, False, r=8),
                BasicBlock(256, 512, True, r=8),
                BasicBlock(512, 512, False, r=8),
            )
            self.dropout = nn.Dropout(p=0.25, inplace=True)
            self.linear = nn.Linear(512, num_classes)
            
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def forward(self, x):
            out = self.conv(x)
            out = F.leaky_relu(self.bn(out))
            out = self.backbone(out)
            out = F.avg_pool2d(out, (4, 4))
            out = out.view(out.shape[0], -1)
            out = self.dropout(out)
            out = self.linear(out)

            return out


    model = MobileResnet()

    return model.to('cuda')

def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.
    
    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    return torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=6e-4)

@torch.no_grad()
def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).
    
    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    return model.forward(batch.to('cuda'))

@torch.no_grad()
def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.
    
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    model.eval()
    loss = accuracy = 0
    bs = dataloader.batch_size
    n = len(dataloader)
    criterion = nn.CrossEntropyLoss()

    for inputs, labels in dataloader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        predictions = predict(model, inputs)
        loss += criterion(predictions, labels)
        accuracy += (labels == torch.argmax(predictions, dim=1)).sum() / bs

    return accuracy / n, loss / n

def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.
    
    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    exp_name = 'experiment'
    n_epochs = 80
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5e-4, eta_min=5e-6, last_epoch=-1)

    os.mkdir(exp_name)
    os.mkdir(exp_name + '/tensorboard')
    writer = SummaryWriter(exp_name + '/tensorboard')

    best_loss = 1e5
    n_train = len(train_dataloader)
    for epoch in range(n_epochs):
        t1 = default_timer()
        loss_train = 0
        model.train()
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            loss_train += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_val, loss_val = validate(val_dataloader, model)
        scheduler.step()

        t2 = default_timer()
        epoch_time = t2 - t1
        writer.add_scalar('time', epoch_time, epoch)
        writer.add_scalar('train_loss', loss_train.item() / n_train, epoch)
        writer.add_scalar('val_acc', acc_val.item(), epoch)
        writer.add_scalar('val_loss', loss_val.item(), epoch)
        writer.add_scalar('l2_norm_linear', model.linear.weight.norm().item(), epoch)

        if loss_val < best_loss:
            best_loss = loss_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, exp_name + '/checkpoint.pth')

def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.
    
    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    md5_checksum = "b8219038eb855f18d147f2faf5977f3f"
    google_drive_link = "https://drive.google.com/file/d/1yY0BoMyW2ZFUhGXMTxCFZyUv6UgJe2tf/view?usp=sharing"
    
    return md5_checksum, google_drive_link