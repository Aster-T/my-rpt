import numpy as np
from keras import ops
from transformers import AutoTokenizer, FlaxAutoModel

from src.constant import embedding_model_to_dimension_and_pooling


class SentenceEmbedder:
    def __init__(self, sentence_embedding_model_name, batch_size=512, device=None):
        super().__init__()
        self.sentence_embedding_model_name = sentence_embedding_model_name
        self.model = FlaxAutoModel.from_pretrained(sentence_embedding_model_name)
        self.embedding_dimension, self.pooling_method = (
            embedding_model_to_dimension_and_pooling[sentence_embedding_model_name]
        )
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(sentence_embedding_model_name)

    def pooling(self, model_output, attention_mask):
        """
        Returns a tensor of shape (batch_size, embedding_dim).
        Supports 'mean' pooling and 'cls' (first token) pooling.
        """
        token_embeddings = model_output[0]  # (batch, seq_len, hidden_dim)
        if self.pooling_method == "mean":
            input_mask_expanded = ops.cast(
                ops.expand_dims(attention_mask, axis=-1),
                dtype=token_embeddings.dtype,
            )
            # Broadcast mask across embedding dim, then masked average
            return ops.sum(token_embeddings * input_mask_expanded, axis=1) / ops.clip(
                ops.sum(input_mask_expanded, axis=1), x_min=1e-9, x_max=None
            )
        assert self.pooling_method == "cls"
        return ops.cast(token_embeddings[:, 0], dtype=token_embeddings.dtype)

    def embed_sentences(self, input_ids, attention_mask):
        """
        Runs batched inference and concatenates pooled embeddings.
        Returns a float16 numpy array of shape (n_sentences, embedding_dim).
        """
        results = []
        for start_idx in range(0, len(input_ids), self.batch_size):
            these_ids = input_ids[start_idx : start_idx + self.batch_size]
            this_mask = attention_mask[start_idx : start_idx + self.batch_size]
            # Flax models return an object; index [0] gives last_hidden_state
            model_output = self.model(
                input_ids=these_ids,
                attention_mask=this_mask,
                train=False,
            )
            results.append(self.pooling(model_output, this_mask))

        # Concatenate all batch results along the sentence axis
        combined = ops.concatenate(results, axis=0)
        # Convert to float16 numpy array, matching PyTorch implementation
        return np.array(combined).astype("float16")

    def embed(self, texts: list[str]):
        """
        Tokenizes a list of strings and returns float16 embeddings
        as a numpy array of shape (len(texts), embedding_dim).
        """
        if not len(texts):
            return []
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np",  # Flax/JAX expects numpy arrays, not pt tensors
            max_length=512,
        )
        embeddings = self.embed_sentences(
            encoded["input_ids"], encoded["attention_mask"]
        )
        return embeddings
