import tensorflow as tf

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate):
        super().__init__()

        # Multi-Head self Attention layer
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = d_model // num_heads,
            dropout = dropout_rate
        )

        # Feed Forward Network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation = "relu"),
            tf.keras.layers.Dense(d_model)
        ])
        
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

        self.drop1 = tf.keras.layers.Dropout(dropout_rate)
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

        
    def call(self, x, training = False):

        # Passing x for Q, K, V because of decoder only architecture we don't have the same 
        # encoder-decoder attention layer so x naturally feels all the place
        # Causal Mask will prevent into peeking to other tokens and 
        # will intimate as Mask Self Attention layer
        attention_output = self.attention(
            query = x,
            key = x, 
            value = x,
            use_causal_mask = True,
            training = training
        )
        attention_output = self.drop1(attention_output, training = training)
        
        # Layer Normalization layer which will help us retain information
        x = self.norm1(attention_output + x)

        ffn = self.ffn(x)
        ffn = self.drop2(ffn, training = training)

        # Information Retention
        x = self.norm2(x + ffn)
        return x

class GPTModel(tf.keras.Model):
    def __init__(self, vocab_size = 95, d_model = 256, max_seq_len = 512,
                num_models = 6, num_heads = 8, dff = 1024, dropout_rate = 0.1):
        super().__init__()

        # Embedding Layer
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim = vocab_size,
            output_dim = d_model
        )

        # Positional Encoding Layer
        self.positional_encoding = tf.keras.layers.Embedding(
            input_dim = max_seq_len,
            output_dim = d_model
        )

        # Constructing 6 DecoderBlock as mentioned in the paper
        self.decoder_block = [DecoderBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_models)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Final layer for decoding the actual word to its output
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x, training = False):
        # Length of the sentence
        seq_len = tf.shape(x)[1]
        positions = tf.range(seq_len)

        # Encoding the embedding with its positional information
        # Original paper of Attention is all you need uses sine and cosine ways
        x = self.embedding_layer(x) + self.positional_encoding(positions)
        
        x = self.dropout(x, training = training)
        for block in self.decoder_block:
            # Passing x to every block (reassigned with change)
            x = block(x, training = training)
        
        return self.output_layer(x)

# Testing
model = GPTModel()
test_input = tf.constant([[72, 69, 76, 76, 79, 1, 87, 79, 82, 76, 68]])
output = model(test_input)
print(f"Output shape: {output.shape}")  
print(f"Trainable vars: {len(model.trainable_variables)}")