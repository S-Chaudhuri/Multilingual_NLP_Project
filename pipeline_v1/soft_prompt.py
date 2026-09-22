"""
Tunable continuous soft prompt embeddings for Fixed-LM Prompt Tuning.

These learnable vectors are prepended to the frozen model's input embeddings.
Only these parameters receive gradient updates during training.

Reference: Lester et al. (2021), "The Power of Scale for Parameter-Efficient
Prompt Tuning".
"""

import torch
import torch.nn as nn


class SoftPromptEmbedding(nn.Module):
    """
    Tunable continuous soft prompt embeddings for Fixed-LM Prompt Tuning.

    These learnable vectors are prepended to the frozen model's input embeddings.
    Only these parameters receive gradient updates during training.
    """

    def __init__(
        self,
        num_prompt_tokens: int,
        embedding_dim: int = 768,
        init_from_vocab: bool = True,
        tokenizer=None,
        model=None,
    ):
        """
        Args:
            num_prompt_tokens: Number of soft prompt tokens (P). E.g., 10, 20, 50, 100.
            embedding_dim: Must match the hidden size of the base model (768 for mBERT base).
            init_from_vocab: If True and model is provided, initialize soft prompts
                from random vocabulary embeddings (recommended by Lester et al.).
                Otherwise, use uniform random initialization.
            tokenizer: The tokenizer (unused but kept for API symmetry with future init strategies).
            model: The pretrained model whose word embeddings are sampled for initialization.
        """
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens  # P (e.g., 10, 20, 50, 100)
        self.embedding_dim = embedding_dim          # 768 for mBERT base

        # Learnable soft prompt parameters
        self.soft_prompt = nn.Parameter(
            torch.empty(num_prompt_tokens, embedding_dim)
        )

        # Initialization strategy
        if init_from_vocab and model is not None:
            self._init_from_vocab(model, tokenizer)
        else:
            nn.init.uniform_(self.soft_prompt, -0.5, 0.5)

    def _init_from_vocab(self, model, tokenizer):
        """Initialize soft prompts from random vocabulary embeddings.

        Sampling from the pretrained embedding table provides a better starting
        point than random uniform initialization (Lester et al., 2021).
        """
        vocab_size = model.config.vocab_size
        random_ids = torch.randint(0, vocab_size, (self.num_prompt_tokens,))
        random_ids = random_ids.to(model.device)
        with torch.no_grad():
            # Access the word embedding layer — works for BERT-family models
            if hasattr(model, 'bert'):
                word_embeddings = model.bert.embeddings.word_embeddings(random_ids)
            elif hasattr(model, 'roberta'):
                word_embeddings = model.roberta.embeddings.word_embeddings(random_ids)
            else:
                # Fallback: try generic get_input_embeddings()
                embed_layer = model.get_input_embeddings()
                word_embeddings = embed_layer(random_ids)
            self.soft_prompt.data.copy_(word_embeddings)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """
        Prepend soft prompt embeddings to input embeddings.

        Args:
            input_embeds: (batch_size, seq_len, embedding_dim) from frozen model embeddings.

        Returns:
            (batch_size, P + seq_len, embedding_dim) with soft prompts prepended.
        """
        batch_size = input_embeds.size(0)
        # Expand soft prompts for the batch: (P, dim) -> (batch_size, P, dim)
        prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        # Concatenate: [soft_prompt | input_embeddings]
        return torch.cat([prompt_embeds, input_embeds], dim=1)

    @property
    def num_trainable_params(self) -> int:
        """Return the number of trainable parameters (P × embedding_dim)."""
        return self.num_prompt_tokens * self.embedding_dim

    def save(self, path: str):
        """Save soft prompt weights to disk."""
        torch.save(self.soft_prompt.data, path)

    @classmethod
    def load(cls, path: str, num_prompt_tokens: int, embedding_dim: int = 768):
        """Load soft prompt weights from disk (no model/tokenizer needed)."""
        instance = cls(
            num_prompt_tokens=num_prompt_tokens,
            embedding_dim=embedding_dim,
            init_from_vocab=False,
        )
        data = torch.load(path, map_location='cpu')
        instance.soft_prompt.data.copy_(data)
        return instance
