"""
training script
"""

import torch

def get_batch(data, batch_size, config):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - config.block_size, (batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters, training_data, val_data, batch_size, config):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == 'train':
                data = training_data
            else:
                data = val_data
            X, Y = get_batch(data, batch_size, config)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
