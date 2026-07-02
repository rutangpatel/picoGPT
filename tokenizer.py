import json
import os
import re
from dataset import import_stories
import numpy as np

class CharTokenizer:
    def __init__(self, text = None, vocab_path = None):
        if vocab_path:
            with open(vocab_path, "r") as f:
                self.char_to_index = json.load(f)
            self.index_to_char = {i : ch for ch, i in self.char_to_index.items()}
            self.vocab_size = len(self.char_to_index)

        elif text:
            self.chars = sorted(set(text))
            self.vocab_size = len(self.chars)
            self.char_to_index = {ch : i for i, ch in enumerate(self.chars)}
            self.index_to_char = {i : ch  for i, ch in enumerate(self.chars)}
        
        else:
            raise ValueError("Provide a path for vocabulary or enter text")

    # Token Encoder block which will convert character into index using vocabulary (a.k.a vocab.json)
    def encode(self, text):
        encoded_output = [self.char_to_index[c] for c in text]
        return encoded_output

    # Token Decoder block which will convert the index to character
    def decode(self, encoded_arr):
        decoded_output = [self.index_to_char[i] for i in encoded_arr]
        return ''.join(decoded_output)

if __name__ == "__main__":
    
    # Build Vocabulary
    data = import_stories(train = True)
    text = re.sub(r'[^\x20-\x7E\n]', '', "".join(data["text"])) # Removing unwanted characters
    token = CharTokenizer(text)

    # Saving vocabulary
    with open("vocab.json", "w") as f:
        json.dump(token.char_to_index, f)
    
    # Warning: This would take long time and might overwhelm the pc, consider running this in colab notebook
    # I ran this on kaggle notebook and it consist of 3.5GB of encoded data so might as well take in consideration
    encoded = np.array(token.encode(text), dtype = np.uint16)
    np.save("encoded_data.npy", encoded)

    # Information about how long the vocabulary is and number of tokens encoded
    print(f"Vocab Size = {token.vocab_size}")
    print(f"Total tokens = {len(encoded)}")

    # Testing the vocabulary with simple example of hello world if it can regenerate it back vocabulary is correct
    token2 = CharTokenizer(vocab_path="vocab.json")
    test = token2.decode(token2.encode("hello world"))
    assert test == "hello world", f"Round trip failed: {test}"
    print("Vocab load verified")