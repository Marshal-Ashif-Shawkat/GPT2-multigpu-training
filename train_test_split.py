import numpy as np

# Load tokens as memory-mapped array (does NOT load into RAM)
tokens = np.memmap("bookcorpus_tokens.bin", dtype=np.uint16, mode="r")

n = len(tokens)
split_idx = int(n * 0.8)

print("Total tokens:", n)
print("Train tokens:", split_idx)
print("Test tokens:", n - split_idx)

# --- Write train ---
with open("train.bin", "wb") as f:
    f.write(tokens[:split_idx].tobytes())

# --- Write test ---
with open("test.bin", "wb") as f:
    f.write(tokens[split_idx:].tobytes())

print("Done.")