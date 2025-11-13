from datasets import load_dataset
import tiktoken
import numpy as np
import time

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

ds = load_dataset("mikasenghaas/wikitext-2")

train_text = "\n".join(ds["train"]["text"])
print("No of characters in the training:", len(train_text))
start_time = time.time()
tokens_np_uint16 = tokenize(train_text)
np.save("train_tokens_wikitext_2.npy", tokens_np_uint16)
end_time = time.time()
print(f"Time taken: {end_time - start_time}")
del train_text
print(tokens_np_uint16.shape)

test_text = "\n".join(ds["test"]["text"])
print("No of characters in the testing:", len(test_text))
start_time = time.time()
tokens_np_uint16 = tokenize(test_text)
np.save("test_tokens_wikitext_2.npy", tokens_np_uint16)
end_time = time.time()
print(f"Time taken: {end_time - start_time}")
del test_text
print(tokens_np_uint16.shape)