import os
import torch
import fire
import numpy as np
from llama import Llama
from datasets import load_dataset
from torch.nn.functional import cross_entropy
from tqdm import tqdm

device = torch.device('cuda')
np.random.seed(69)

def get_batch(data, max_seq_len, max_batch_size):
    ix = torch.randint(len(data) - max_seq_len, (max_batch_size,))
    # x is the input sequence, y is the target sequence which is x shifted by 1
    x = torch.stack([torch.from_numpy((data[i:i+max_seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+max_seq_len]).astype(np.int64)) for i in ix])
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

@torch.inference_mode()
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    dataset: str,
    eval_iters: int = 100,
    max_seq_len: int = 4096,
    max_batch_size: int = 1,
    last_k: int = None,
    fade: bool = False,
):
    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        fade=fade,
    )

    if os.path.exists('data/' + dataset + '/test.bin'):
        data = np.fromfile('data/' + dataset + '/test.bin', dtype=np.uint16)
    else:
        # throw error
        print("test.bin file not found")
        return
    
    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    generate_times = []
    forward_times = []

    model = llama.model
    model.eval()

    n_gen_tokens = 1
    losses = torch.zeros(eval_iters)
    for k in tqdm(range(eval_iters), desc='Evaluating'):
        X, Y = get_batch(data, max_seq_len, max_batch_size)
        # time generation with pytorch cuda events
        torch.cuda.synchronize()
        start_event.record()

        pred = llama.generate(prompt_tokens=X, max_gen_len=n_gen_tokens)

        end_event.record()
        torch.cuda.synchronize()
        # compute the time in milliseconds
        generate_times.append(start_event.elapsed_time(end_event)/n_gen_tokens)

        # time generation with pytorch cuda events
        torch.cuda.synchronize()
        start_event.record()

        logits = model.forward(X, 0)

        end_event.record()
        torch.cuda.synchronize()
        # compute the time in milliseconds
        forward_times.append(start_event.elapsed_time(end_event))

        # calculate loss for only the last_k tokens
        if last_k is not None:
            logits = logits[:, -last_k:, :]
            Y = Y[:, -last_k:]
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), Y.reshape(-1), ignore_index=-1)
        losses[k] = loss.item()

    avg_loss = losses.mean().item()
    avg_perplexity = torch.exp(losses).mean().item()
    avg_generate_time = np.mean(generate_times)
    avg_forward_time = np.mean(forward_times)

    print('model: ', ckpt_dir, 'dataset: ', dataset)
    print(f'avg loss: {avg_loss:.4f}')
    print(f'avg perplexity: {avg_perplexity:.4f}')
    print(f'avg generate time: {avg_generate_time*1000:.4f}ns/token')
    print(f'avg forward time: {avg_forward_time:.4f}ms')

if __name__ == '__main__':
    fire.Fire(main)