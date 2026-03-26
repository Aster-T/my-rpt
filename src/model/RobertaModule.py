from keras.layers import Dense, Dropout, Layer, LayerNormalization


class RobertaIntermediate(Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = Dense(config.intermediate_dim, activation=config.activation)

    def call(self, hidden_states):
        return self.dense(hidden_states)


class RobertaOutput(Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = Dense(config.hidden_dim)
        self.layer_norm = LayerNormalization(epsilon=config.layer_norm_eps)
        self.dropout = Dropout(config.dropout)

    def call(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class RobertaSelfOutput(Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = Dense(config.hidden_dim)
        self.LayerNorm = LayerNormalization(epsilon=config.layer_norm_eps)
        self.dropout = Dropout(config.dropout)

    def call(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
