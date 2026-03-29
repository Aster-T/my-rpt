import numpy as np

try:
    from keras import backend as keras_backend
    from keras import ops
except ImportError:
    keras_backend = None
    ops = None
from transformers import AutoTokenizer

from src.constant import embedding_model_to_dimension_and_pooling


class SentenceEmbedder:
    def __init__(
        self, sentence_embedding_model_name: str, batch_size: int = 512, device=None
    ):
        super().__init__()
        if (
            sentence_embedding_model_name
            not in embedding_model_to_dimension_and_pooling
        ):
            supported_models = ", ".join(
                sorted(embedding_model_to_dimension_and_pooling.keys())
            )
            raise ValueError(
                "Unsupported sentence embedding model "
                f"{sentence_embedding_model_name!r}. "
                f"Supported models: {supported_models}."
            )

        if ops is None or keras_backend is None:
            raise ImportError(
                "SentenceEmbedder requires Keras 3 with the JAX backend installed."
            )
        if keras_backend.backend() != "jax":
            raise RuntimeError(
                "SentenceEmbedder requires the Keras backend to be 'jax' because it "
                "wraps a Flax model."
            )

        self.sentence_embedding_model_name = sentence_embedding_model_name
        try:
            from transformers import FlaxAutoModel
        except ImportError as exc:
            raise ImportError(
                "SentenceEmbedder requires transformers with Flax support. "
                "Install compatible 'transformers', 'flax', and 'jax' packages."
            ) from exc

        self.model = FlaxAutoModel.from_pretrained(sentence_embedding_model_name)
        self.embedding_dimension, self.pooling_method = (
            embedding_model_to_dimension_and_pooling[sentence_embedding_model_name]
        )
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(sentence_embedding_model_name)
        self.device = device

    def pooling(self, model_output, attention_mask):
        """
        Returns a numpy array of shape (batch_size, embedding_dim).
        Supports 'mean' pooling and 'cls' (first token) pooling.
        """
        token_embeddings = model_output[0]  # (batch, seq_len, hidden_dim)
        if self.pooling_method == "mean":
            input_mask_expanded = ops.cast(
                ops.expand_dims(attention_mask, axis=-1),
                dtype=token_embeddings.dtype,
            )
            pooled = ops.sum(token_embeddings * input_mask_expanded, axis=1)
            denominator = ops.clip(
                ops.sum(input_mask_expanded, axis=1), x_min=1e-9, x_max=None
            )
            return pooled / denominator

        if self.pooling_method != "cls":
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")
        return token_embeddings[:, 0]

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
        return np.asarray(combined, dtype=np.float16)

    def embed(self, texts: list[str]):
        """
        Tokenizes a list of strings and returns float16 embeddings
        as a numpy array of shape (len(texts), embedding_dim).
        """
        if not len(texts):
            return np.empty((0, self.embedding_dimension), dtype=np.float16)
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
