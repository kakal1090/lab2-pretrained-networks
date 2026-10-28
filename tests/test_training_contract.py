"""Meaningful training checks without network or real dataset."""
import unittest
from unittest.mock import patch
from argparse import Namespace
import torch
from src.train import build_model, training_mode, run_epoch, make_loaders
from torch.utils.data import DataLoader, TensorDataset


class TrainingContract(unittest.TestCase):
    def test_frozen_backbone_and_head_update(self):
        torch.set_num_threads(2)
        for name in ('resnet18', 'densenet121'):
            with self.subTest(model=name):
                model = build_model(name, 'head', pretrained=False)
                head = model.fc if name == 'resnet18' else model.classifier
                head_before = head.weight.detach().clone()
                frozen = {n: v.clone() for n,v in model.state_dict().items()
                          if not n.startswith('fc.') and not n.startswith('classifier.')}
                training_mode(model, name, 'head')
                opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.1)
                loader = DataLoader(TensorDataset(torch.randn(2,3,64,64), torch.tensor([1,2])), batch_size=2)
                metrics = run_epoch(model, loader, torch.device('cpu'), opt)
                self.assertEqual(metrics['samples'], 2)
                self.assertFalse(torch.equal(head_before, head.weight))
                for n, old in frozen.items():
                    self.assertTrue(torch.equal(old, model.state_dict()[n]), n)

    def test_stratified_split_no_leakage(self):
        class DummyCIFAR:
            targets = [label for label in range(10) for _ in range(5000)]
            def __len__(self):
                return len(self.targets)
            def __getitem__(self, index):
                return torch.zeros(3, 224, 224), self.targets[index]
        args = Namespace(smoke=False, data_dir='unused', seed=42, batch_size=32, workers=0)
        with patch('src.train.datasets.CIFAR10', return_value=DummyCIFAR()):
            loaders, _, split = make_loaders(args)
            _, _, repeat = make_loaders(args)
        train, val = set(split['train_indices']), set(split['val_indices'])
        self.assertEqual(len(train), 45000)
        self.assertEqual(len(val), 5000)
        self.assertFalse(train & val)
        self.assertEqual(len(train | val), 50000)
        self.assertEqual(split, repeat)
        for label in range(10):
            self.assertEqual(sum(DummyCIFAR.targets[i] == label for i in val), 500)

    def test_finetune_scope_and_batchnorm_mode(self):
        model = build_model('resnet18', 'finetune', pretrained=False)
        self.assertTrue(all(n.startswith(('layer4.', 'fc.')) for n,p in model.named_parameters() if p.requires_grad))
        self.assertTrue(any(n.startswith('layer4.') for n,p in model.named_parameters() if p.requires_grad))
        training_mode(model, 'resnet18', 'finetune')
        self.assertFalse(model.layer1.training)
        self.assertTrue(model.layer4.training)
        self.assertTrue(model.fc.training)


if __name__ == '__main__':
    unittest.main()
