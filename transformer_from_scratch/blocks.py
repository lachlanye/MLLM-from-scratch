# transformer_from_scratch/blocks.py

import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .layers import PositionwiseFeedForward


class EncoderBlock(nn.Module):
    """
    Transformer Encoder 的基本构建块。
    包含一个多头自注意力层和一个位置前馈网络。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # --- YOUR CODE HERE ---
        # TODO: 实例化多头自注意力层和位置前馈网络层。
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # --- END YOUR CODE ---

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): EncoderBlock的输入, shape [B, L_src, d_model]
            src_mask (torch.Tensor): 源序列的掩码

        Returns:
            torch.Tensor: EncoderBlock的输出, shape [B, L_src, d_model]
        """
        # --- YOUR CODE HERE ---
        # TODO: 实现 EncoderBlock 的前向传播 (Pre-LN)

        # 1. Pre-LN Self-Attention
        #    x = x + attn(norm(x))
        src2 = self.norm1(src)
        src = src + self.self_attn(src2, src2, src2, src_mask)

        # 2. Pre-LN Feed-Forward
        #    x = x + ffn(norm(x))
        src2 = self.norm2(src)
        src = src + self.feed_forward(src2)

        return src
        # --- END YOUR CODE ---


class DecoderBlock(nn.Module):
    """
    Transformer Decoder 的基本构建块。
    包含一个掩码多头自注意力层、一个多头交叉注意力层和一个位置前馈网络。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # --- YOUR CODE HERE ---
        # TODO: 实例化三个核心组件
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # --- END YOUR CODE ---

    def forward(self, tgt: torch.Tensor, enc_src: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt (torch.Tensor): DecoderBlock的输入, shape [B, L_tgt, d_model]
            enc_src (torch.Tensor | None): Encoder的输出, shape [B, L_src, d_model]。
            tgt_mask (torch.Tensor | None): 目标序列的掩码 (用于自注意力)
            src_mask (torch.Tensor | None): 源序列的掩码 (用于交叉注意力)

        Returns:
            torch.Tensor: DecoderBlock的输出, shape [B, L_tgt, d_model]
        """
        # --- YOUR CODE HERE ---
        # TODO: 实现 DecoderBlock 的前向传播 (Pre-LN)

        # 1. Pre-LN Masked Self-Attention
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.self_attn(tgt2, tgt2, tgt2, tgt_mask)

        # 只有在 enc_src (Encoder的输出) 被提供时，才执行交叉注意力。
        if enc_src is not None:
            # 2. Pre-LN Cross-Attention
            tgt2 = self.norm2(tgt)
            tgt = tgt + self.cross_attn(tgt2, enc_src, enc_src, src_mask)

        # 3. Pre-LN Feed-Forward
        tgt2 = self.norm3(tgt)
        tgt = tgt + self.feed_forward(tgt2)

        return tgt
        # --- END YOUR CODE ---
