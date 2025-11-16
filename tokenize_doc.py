from datasets import load_dataset
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np

ds = load_dataset("rojagtap/bookcorpus", split="train", streaming=True)

outfile = open("bookcorpus_tokens.bin", "wb")

count = 0
for item in ds:
    tokens = tokenize(item["text"])
    outfile.write(tokens.tobytes())
    count += len(tokens)
    if count % 1_000_000 == 0:
        print(f"{count} tokens written...")

outfile.close()
print("Done!")