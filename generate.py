import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from model import GPTModel
from tokenizer import CharTokenizer

SEQ_LEN = 128

def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0):
    # Encode prompt and ensure 2D shape: (1, prompt_length)
    input_ids = tokenizer.encode(prompt)
    input_tensor = tf.constant([input_ids], dtype=tf.int32)  # shape: (1, len)
    
    # Track only newly generated tokens for clean output
    generated_tokens = []
    
    for _ in range(max_new_tokens):
        # Crop to max_seq_len BEFORE passing to model
        # We keep the last SEQ_LEN tokens as context
        if input_tensor.shape[1] > SEQ_LEN:
            context = input_tensor[:, -SEQ_LEN:]  # shape: (1, SEQ_LEN)
        else:
            context = input_tensor  # shape: (1, current_len)
        
        # Forward pass on cropped context only
        logits = model(context, training=False)  # shape: (1, seq_len, vocab_size)
        
        # Get logits for the LAST position of the context
        next_token_logits = logits[0, -1, :]  # shape: (vocab_size,)
        next_token_logits = next_token_logits / temperature
        
        # Sample from distribution
        probs = tf.nn.softmax(next_token_logits)
        next_token = tf.random.categorical(
            tf.math.log(probs[tf.newaxis, :]),
            num_samples=1
        )[0, 0]
        
        next_token = tf.cast(next_token, tf.int32)
        
        # Append to full history (this grows beyond SEQ_LEN, which is fine)
        input_tensor = tf.concat([input_tensor, [[next_token]]], axis=1)
        generated_tokens.append(int(next_token))
    
    # Decode only the generated part (not the prompt)
    return tokenizer.decode(generated_tokens)

def main():
    tokenizer = CharTokenizer(vocab_path="vocab.json")
    
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        max_seq_len=SEQ_LEN,
        num_models=6,
        num_heads=8,
        dff=1024,
        dropout_rate=0.1
    )
    
    # Build model before loading weights
    dummy = tf.constant([[0]])
    _ = model(dummy)
    
    model.load_weights("picoGPT.weights.h5")
    print("Weights Loaded")
    
    # Test prompts
    prompts = [
        "Once upon a time"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        
        # Generate and print prompt + generation together
        generated = generate(model, tokenizer, prompt, max_new_tokens=500)
        print(prompt + generated)

if __name__ == "__main__":
    main()