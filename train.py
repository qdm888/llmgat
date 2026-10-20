import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data import get_dataloaders
from models import LLM_GAT, modality_consistency_loss
from utils import eval_regression

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('llm_gat')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def joint_loss(outputs, labels, cls_criterion, lam):
    logits, aux = outputs
    l_cls = cls_criterion(logits, labels.view(-1, 1))
    l_cons = modality_consistency_loss(aux)
    return l_cls + lam * l_cons


def evaluate(model, loader, device, dataset, return_attn=False):
    model.eval()
    y_pred, y_true, ids = [], [], []
    attn_store = []
    with torch.no_grad():
        for batch in loader:
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            outputs = model(text, audio, vision, return_attn=return_attn)
            y_pred.append(outputs[0].view(-1).cpu())
            y_true.append(batch['label'].view(-1))
            ids.extend(batch['id'])
            if return_attn:
                attn_store.append(outputs[1]['edge_attn'].mean(dim=1).cpu())
    y_pred = torch.cat(y_pred).numpy()
    y_true = torch.cat(y_true).numpy()
    results = eval_regression(y_pred, y_true, dataset)
    if return_attn:
        results['edge_attn'] = torch.cat(attn_store)
        results['ids'] = ids
    return results


def train(args):
    cfg = json.loads(Path(args.config).read_text())
    cfg['dataset_name'] = args.dataset
    if args.fusion_mode:
        cfg['fusion_mode'] = args.fusion_mode
    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                          and args.gpu >= 0 else 'cpu')

    loaders = get_dataloaders(args.feature_path, cfg['batch_size'],
                              num_workers=args.num_workers)
    model = LLM_GAT(cfg).to(device)
    logger.info(f'trainable parameters: '
                f'{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    cls_criterion = nn.MSELoss()
    lam = cfg.get('consistency_weight', 0.1)

    best_valid, best_epoch = 1e8, 0
    save_path = Path(args.model_save_dir) / f'llm_gat-{args.dataset}.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg['max_epochs'] + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(loaders['train'], desc=f'epoch {epoch}'):
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            loss = joint_loss(model(text, audio, vision), labels,
                              cls_criterion, lam)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(loaders['train'])

        val_results = evaluate(model, loaders['valid'], device, args.dataset)
        logger.info(f'epoch {epoch} | train loss {train_loss:.4f} | '
                    f'valid {val_results}')

        if val_results['MAE'] < best_valid - 1e-6:
            best_valid, best_epoch = val_results['MAE'], epoch
            torch.save(model.state_dict(), save_path)
        if epoch - best_epoch >= cfg['early_stop']:
            break

    model.load_state_dict(torch.load(save_path))
    test_results = evaluate(model, loaders['test'], device, args.dataset)
    logger.info(f'TEST {test_results}')
    return test_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['mosi', 'mosei', 'sims'])
    parser.add_argument('--feature_path', required=True,
                        help='feature pickle (use the LLM-DA augmented file '
                             'to enable augmentation)')
    parser.add_argument('--config', default=None,
                        help='config json; defaults to configs/<dataset>.json')
    parser.add_argument('--fusion_mode', default=None,
                        choices=['gat', 'concat'],
                        help='override the fusion module (concat = ablation)')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_save_dir', default='saved_models')
    args = parser.parse_args()
    if args.config is None:
        args.config = str(Path(__file__).parent / 'configs'
                          / f'{args.dataset}.json')
    train(args)


if __name__ == '__main__':
    main()
