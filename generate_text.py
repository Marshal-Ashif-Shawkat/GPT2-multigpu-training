import tiktoken
import torch

from model import config, GPT

def generate_text_simple(model, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Get the predictions
        with torch.no_grad():
            logits = model(idx)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


checkpoint_file_path = "gpt2_wikitext2.pt"
device = 'cuda'
model = GPT(config).to(device)
model = torch.compile(model)
model.load_state_dict(torch.load(checkpoint_file_path, map_location=device, weights_only=True))

input_prompt = "The capital city"

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(input_prompt, tokenizer).to(device),
    max_new_tokens=25
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))