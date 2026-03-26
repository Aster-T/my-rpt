from keras import ops
from keras.layers import Dense, Dropout, Layer
from keras_hub.models import RobertaBackbone

from .RobertaModule import RobertaIntermediate, RobertaOutput, RobertaSelfOutput


class TwoDimensionAttentionLayer(Layer):
    def __init__(self, config):
        super().__init__()
        layer_class = RobertaBackbone
        self.cross_column_layer = layer_class(**config.to_kwargs(layer_class))
        self.cross_row_layer = layer_class(**config.to_kwargs(layer_class))

    def call(self, hidden_states, attention_mask=None):
        """
        hidden_states: shape (num_rows, num_columns, hidden_size)
        attention_mask: shape (num_rows, num_rows) and values True (attend) or False (do not attend).
        Applies the two separate RobertaLayers, once along the "columns" direction and once along the "rows" direction.
        Along the columns direction, the attention is always full.
        Along the rows direction, an attention mask should be provided (to avoid that context rows can attend query rows)

        Returns: tensor of shape (num_rows, num_columns, hidden_size)
        """

        num_rows, num_columns, _ = ops.shape(hidden_states)

        max_row_per_batch = 8192
        col_fraction = 100.0 / float(num_columns)
        batch_step = int(max_row_per_batch * col_fraction)

        # horizontal attention (full attention along columns)
        horizontal_outputs = []
        for i in range(0, num_rows, batch_step):
            chunk = hidden_states[i : i + batch_step, :, :]
            chunk_output = self.cross_column_layer(chunk)
            horizontal_outputs.append(chunk_output)
        horizontal_output = ops.concatenate(horizontal_outputs, axis=0)

        batch_step = 100
        # attention_mask (num_rows, num_rows) -> (1, 1, num_rows, num_rows)
        attention_mask = ops.expand_dims(attention_mask, axis=0)
        attention_mask = ops.expand_dims(attention_mask, axis=0)

        # vertical attention (attention along rows, with attention mask)
        horizontal_output = ops.transpose(horizontal_output, axes=(1, 0, 2))
        vertical_outputs = []
        for i in range(0, num_columns, batch_step):
            chunk = horizontal_output[i : i + batch_step, :, :]
            chunk_output = self.cross_row_layer(chunk, attention_mask)
            vertical_outputs.append(chunk_output)
        vertical_output = ops.concatenate(vertical_outputs, axis=0)

        return ops.transpose(vertical_output, axes=(1, 0, 2))


class KerasRobertaLayer(Layer):
    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = KerasAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def call(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class KerasAttention(Layer):
    def __init__(self, config):
        super().__init__()
        self.self_attention = KerasSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def call(self, hidden_states, attention_mask=None):
        self_outputs = self.self_attention(hidden_states, attention_mask)
        return self.output(self_outputs, hidden_states)


class KerasSelfAttention(Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.hidden_dim % config.num_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_dim}) is not a multiple of the number of attention heads ({config.num_heads})"
            )

        self.num_attention_heads = config.num_heads
        assert config.hidden_dim % self.num_attention_heads == 0, (
            f"{config.hidden_dim=} must be divisible by {config.num_attention_heads=}"
        )
        self.attention_head_size = config.hidden_dim // self.num_attention_heads

        # Q K V projection layers
        self.query = Dense(config.hidden_dim)
        self.key = Dense(config.hidden_dim)
        self.value = Dense(config.hidden_dim)

        self.dropout = Dropout(config.dropout)

    def transpose_for_scores(self, x):
        batch_size, seq_len = ops.shape(x)
        x = ops.reshape(
            x, (batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        )
        return ops.transpose(x, axes=(0, 2, 1, 3))

    def call(self, hidden_states, attention_mask=None):
        """

        hidden_states: shape (batch_size, seq_len, hidden_size)
        attention_mask: shape (batch_size, seq_len, seq_len) or (batch_size, 1, seq_len, seq_len)
        attention_mask should be a boolean mask where True indicates positions to attend to and False indicates positions to mask out.
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attn_output = ops.dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            mask=attention_mask,
        )
        attn_output = self.dropout(attn_output)

        context_layer = ops.transpose(attn_output, axes=(0, 2, 1, 3))
        batch_size, seq_len, _ = ops.shape(context_layer)
        context_layer = ops.reshape(
            context_layer, (batch_size, seq_len, self.config.hidden_dim)
        )
        return context_layer
