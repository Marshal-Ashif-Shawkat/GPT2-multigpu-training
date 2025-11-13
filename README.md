# GPT2 multigpu training using PyTorch DDP

Inspired by the lecture video of [Andrej Karpathy](https://youtu.be/l8pRSuU81PU?si=JNiBO_nSUvNt-dG0) and [Sebastian Raschka](https://youtu.be/yAcWnfsZhzo?si=e3x6e5a9AiFGbq9k), I decided to build and train my own GPT-2 model. Due to the GPU limitation, I rented 4x Nvidia RTX 5070 Ti GPU from Vast.ai. As a proof-of-concept, I use the [Wikitext-2](https://huggingface.co/datasets/mikasenghaas/wikitext-2) dataset from HuggingFace to train the model.

Assumptions: You have already installed PyTorch in your environment. You have access to a single node multi-gpu setup. Your gpus support TensorCore.  
Requirements: PyTorch >= 2.0  
How to use
1. run tokenize_doc.py to create the tokens. Two numpy file will be saved in your current directory.
2. Then run gpt2_multigpu.py using torchrun
3. After finisihing the model training, run generate_text.py to check how well your model is generating text.

This is a proof-of-concept. It is not heavily tested.