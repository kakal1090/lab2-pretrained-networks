"""Practice 2: reproducible CIFAR-10 transfer learning. Run from repo root."""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import torchvision


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(name, mode, pretrained=True):
    if name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 10)
        if mode == 'finetune':
            for p in model.layer4.parameters():
                p.requires_grad = True
    else:
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, 10)
        if mode == 'finetune':
            for block in (model.features.denseblock4, model.features.norm5):
                for p in block.parameters():
                    p.requires_grad = True
    return model


def training_mode(model, name, mode):
    # Keep frozen BatchNorm running statistics fixed as well as weights.
    model.eval()
    if name == 'resnet18':
        model.fc.train()
        if mode == 'finetune':
            model.layer4.train()
    else:
        model.classifier.train()
        if mode == 'finetune':
            model.features.denseblock4.train()
            model.features.norm5.train()


def make_loaders(args):
    # Same deterministic transform for both architectures; no augmentation confound.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if args.smoke:
        train = datasets.FakeData(size=4, image_size=(3, 224, 224), num_classes=10,
                                  transform=transforms.ToTensor())
        val = datasets.FakeData(size=2, image_size=(3, 224, 224), num_classes=10,
                                transform=transforms.ToTensor(), random_offset=100)
        split = {'synthetic': True, 'train_size': 4, 'val_size': 2}
    else:
        full = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform)
        # Stratified 90/10 split; fixed seed, independent of model and batch size.
        rng = random.Random(args.seed)
        train_indices, val_indices = [], []
        for label in range(10):
            indices = [i for i, target in enumerate(full.targets) if target == label]
            rng.shuffle(indices)
            val_indices.extend(indices[:500])
            train_indices.extend(indices[500:])
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        train, val = Subset(full, train_indices), Subset(full, val_indices)
        split = {'seed': args.seed, 'train_indices': train_indices, 'val_indices': val_indices}
    generator = torch.Generator().manual_seed(args.seed)
    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=torch.cuda.is_available())
    loaders = {
        'train': DataLoader(train, shuffle=True, generator=generator, **common),
        'val': DataLoader(val, shuffle=False, **common),
    }
    return loaders, transform, split


def run_epoch(model, loader, device, optimizer=None):
    loss_sum, correct, count = 0., 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            count += y.size(0)
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    return {'loss': loss_sum / count, 'accuracy': correct / count, 'samples': count}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['resnet18', 'densenet121'], default='resnet18')
    p.add_argument('--mode', choices=['head', 'finetune'], default='head')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output-root', default='results')
    p.add_argument('--run-name', required=True)
    p.add_argument('--smoke', action='store_true', help='Synthetic pipeline check; NOT lab results')
    p.add_argument('--evaluate-test', action='store_true', help='Only after selecting configuration on validation')
    p.add_argument('--init-checkpoint', help='Optional baseline best.pt for warm-start fine-tuning')
    args = p.parse_args()
    if args.lr <= 0 or args.batch_size < 1 or args.epochs < 1:
        p.error('lr, batch-size and epochs must be positive')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        p.error('run-name must be a simple folder name')
    if args.smoke and args.init_checkpoint:
        p.error('Do not mix a real checkpoint with synthetic smoke testing')
    return args


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.output_root) / args.run_name
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous experiment.
    config = vars(args) | {'device': str(device), 'python': platform.python_version(),
                           'torch': torch.__version__, 'torchvision': torchvision.__version__,
                           'pretrained': not args.smoke, 'optimizer': 'SGD', 'momentum': 0.9}
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    model = build_model(args.model, args.mode, pretrained=not args.smoke).to(device)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state['model'])
    (out / 'architecture.txt').write_text(str(model), encoding='utf-8')
    params = {'total': sum(p.numel() for p in model.parameters()),
              'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'trainable_names': [n for n, p in model.named_parameters() if p.requires_grad]}
    (out / 'parameters.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
    print(f'Device={device}; parameters={params["total"]}; trainable={params["trainable"]}')
    loaders, transform, split = make_loaders(args)
    (out / 'split.json').write_text(json.dumps(split), encoding='utf-8')
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                                lr=args.lr, momentum=0.9)
    writer = SummaryWriter(str(out / 'tensorboard'))
    best_acc, best_epoch = -1., 0
    start = time.perf_counter()
    rows = []
    try:
        for epoch in range(1, args.epochs + 1):
            training_mode(model, args.model, args.mode)
            train = run_epoch(model, loaders['train'], device, optimizer)
            model.eval()
            val = run_epoch(model, loaders['val'], device)
            row = {'epoch': epoch, 'train_loss': train['loss'], 'train_accuracy': train['accuracy'],
                   'val_loss': val['loss'], 'val_accuracy': val['accuracy']}
            rows.append(row)
            for key, value in row.items():
                if key != 'epoch':
                    writer.add_scalar(key, value, epoch)
            writer.flush()
            with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerows(rows)
            if val['accuracy'] > best_acc:
                best_acc, best_epoch = val['accuracy'], epoch
                torch.save({'model': model.state_dict(), 'epoch': epoch}, out / 'best.pt')
            print(row, flush=True)
        summary = {'status': 'synthetic_smoke_only' if args.smoke else 'completed',
                   'best_val_accuracy': best_acc, 'best_epoch': best_epoch,
                   'training_seconds': time.perf_counter() - start, 'test': None}
        if args.evaluate_test and not args.smoke:
            model.load_state_dict(torch.load(out / 'best.pt', map_location=device,
                                              weights_only=True)['model'])
            model.eval()
            testset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform)
            testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.workers)
            summary['test'] = run_epoch(model, testloader, device)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
    finally:
        writer.close()


if __name__ == '__main__':
    main()
