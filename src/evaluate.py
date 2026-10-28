"""Evaluate saved best checkpoint, after validation-based configuration selection."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .train import build_model, run_epoch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    folder = Path(args.run_dir)
    cfg = json.loads((folder / 'config.json').read_text())
    if cfg['smoke']:
        raise ValueError('Synthetic run cannot be evaluated as lab results')
    target = folder / 'test_metrics.json'
    if target.exists():
        raise FileExistsError('Test already evaluated. Use the existing test_metrics.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg['model'], cfg['mode'], pretrained=False).to(device)
    model.load_state_dict(torch.load(folder / 'best.pt', map_location=device, weights_only=True)['model'])
    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    ds = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform)
    metrics = run_epoch(model, DataLoader(ds, batch_size=cfg['batch_size'], num_workers=args.workers), device)
    target.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(metrics)


if __name__ == '__main__':
    main()
