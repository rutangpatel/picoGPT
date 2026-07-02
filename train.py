import os
import json
import tensorflow as tf
import numpy as np

from model import GPTModel
from tokenizer import CharTokenizer

# Hyperparameters 
# Can ignore this step cause parameters are already defined in the GPTModel's
# __init__ method expcept EPOCHS and BATCH_SIZE cause they are not define in GPTModel
SEQ_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
D_MODEL = 256
NUM_HEADS = 8
NUM_MODELS = 6
DFF = 1024
DROPOUT = 0.1

def create_dataset(encoded_train_data, seq_len, batch_size):
    # Converting numpy array into Tensor for faster computation
    dataset = tf.data.Dataset.from_tensor_slices(encoded_train_data)

    # Window creation for input and target
    # Our window is [0, 1, 2, 3, 4] we will split into,
    # input = [0, 1, 2, 3] and the job of the model will
    # predict the target = [1, 2, 3, 4]
    dataset = dataset.window(seq_len + 1, shift = 1, drop_remainder = True)
    
    # Due to window function dataset has converted from tensors to nested dataset
    # This will help bring into tensors
    dataset = dataset.flat_map(lambda w : w.batch(seq_len + 1))

    # For seperation of input and target
    def split(window):
        return window[:-1], window[1:]
    
    # Tensor will go from size (129,) to ((128,), (128,))
    dataset = dataset.map(split)
    
    # Shuffle the dataset
    dataset = dataset.shuffle(10000)

    # Batch 32 examples together
    dataset = dataset.batch(batch_size)

    # For optimization of GPU and CPU so they don't sit idle
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def main():
    # Load tokenizer
    tokenizer = CharTokenizer(vocab_path = "vocab.json")
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Load encoded data
    encoded = np.load("encoded_train_data.npy")
    print(f"Total tokens: {len(encoded)}")
    
    # Dataset Creation
    train_ds = create_dataset(encoded, SEQ_LEN, BATCH_SIZE)

    # Build Model
    model = GPTModel(
        vocab_size = vocab_size,
        d_model = D_MODEL,
        max_seq_len = SEQ_LEN,
        num_models = NUM_MODELS,
        num_heads = NUM_HEADS,
        dff = DFF,
        dropout_rate = DROPOUT
    )

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-4),
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ["accuracy"])
    
    # Training
    print("Training Started....")
    history = model.fit(train_ds, epochs = EPOCHS)

    # Saving weights
    model.save_weights("picoGPT.weights.h5")
    print("Model Saved")

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = main()