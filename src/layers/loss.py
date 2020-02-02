import torch
from utils import to_var


def masked_cross_entropy(logits, target, length, per_example=False):
    """
    Source: https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
    """
    batch_size, max_len, num_classes = logits.size()

    # [batch_size * max_len, num_classes]
    logits_flat = logits.view(-1, num_classes)

    # [batch_size * max_len, num_classes]
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=1)

    # [batch_size * max_len, 1]
    target_flat = target.view(-1, 1)

    # Negative Log-likelihood: -sum {  1* log P(target)  + 0 log P(non-target)} = -sum( log P(target) )
    # [batch_size * max_len, 1]
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # [batch_size, max_len]
    losses = losses_flat.view(batch_size, max_len)

    # [batch_size, max_len]
    mask = sequence_mask(sequence_length=length, max_len=max_len)

    # Apply masking on loss
    losses = losses * mask.float()

    # word-wise cross entropy
    # loss = losses.sum() / length.float().sum()

    if per_example:
        # loss: [batch_size]
        return losses.sum(1)
    else:
        loss = losses.sum()
        return loss, length.float().sum()


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.max()
    batch_size = sequence_length.size(0)

    seq_range = torch.arange(0, max_len).long()

    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = to_var(seq_range_expand)

    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)

    masks = seq_range_expand < seq_length_expand

    return masks
