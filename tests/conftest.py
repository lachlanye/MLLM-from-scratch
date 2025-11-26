import pytest
import torch


@pytest.fixture
def params():
    """提供 attention 测试复用的输入张量。"""
    d_model = 128
    n_heads = 8
    batch_size = 4
    seq_len = 10
    d_k = d_model // n_heads

    src_len = 12
    tgt_len = 10
    src_vocab_size = 100
    tgt_vocab_size = 120

    src_ids = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt_ids = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    src_embed = torch.randn(batch_size, src_len, d_model)
    tgt_embed = torch.randn(batch_size, tgt_len, d_model)
    enc_output = torch.randn(batch_size, src_len, d_model)

    # Masks based on IDs
    src_mask = (src_ids != 0).unsqueeze(1).unsqueeze(2)

    tgt_padding_mask = (tgt_ids != 0).unsqueeze(1).unsqueeze(2)
    tgt_causal_mask = torch.tril(torch.ones(
        tgt_len, tgt_len, dtype=torch.bool))
    tgt_mask = tgt_padding_mask & tgt_causal_mask

    return {
        'batch_size': batch_size,
        'd_model': d_model,
        'n_heads': n_heads,
        'seq_len': seq_len,
        'd_k': d_k,
        'd_ff': d_model * 4,
        'dropout': 0.1,
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
        'num_layers': 2,
        'max_len': 100,
        'pad_idx': 0,
        'src_len': src_len,
        'tgt_len': tgt_len,
        'query_sdpa': torch.randn(batch_size, n_heads, seq_len, d_k),
        'key_sdpa': torch.randn(batch_size, n_heads, seq_len, d_k),
        'value_sdpa': torch.randn(batch_size, n_heads, seq_len, d_k),
        'query_mha': torch.randn(batch_size, seq_len, d_model),
        'key_mha': torch.randn(batch_size, seq_len, d_model),
        'value_mha': torch.randn(batch_size, seq_len, d_model),
        'src_ids': src_ids,
        'tgt_ids': tgt_ids,
        'src_embed': src_embed,
        'tgt_embed': tgt_embed,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'enc_output': enc_output,
    }
