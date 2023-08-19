import os
import np
import torch
import fire
from generation import Llama
from datasets import load_dataset

device = torch.device('cuda')

def get_batch(data, max_seq_len, max_batch_size):
    ix = torch.randint(len(data) - max_seq_len, (max_batch_size,))
    # x is the input sequence, y is the target sequence which is x shifted by 1
    x = torch.stack([torch.from_numpy((data[i:i+max_seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+max_seq_len]).astype(np.int64)) for i in ix])
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    dataset: str,
    eval_iters: int = 100,
    max_seq_len: int = 4096,
    max_batch_size: int = 1,
    last_k: int = None,
):
    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    if os.path.exists('data/' + dataset + '/test.bin'):
        data = np.fromfile('data/' + dataset + '/test.bin', dtype=np.uint16)
    else:
        # throw error
        print("test.bin file not found")
        return
    
    l# Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    generate_times = []

    model = llama.model
    model.eval()

    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, max_seq_len, max_batch_size)
        # time generation with pytorch cuda events
        torch.cuda.synchronize()
        start_event_generate.record()

        pred = llama.generate(prompt_tokens=X, max_gen_len=1)

        end_event_generate.record()
        torch.cuda.synchronize()
        # compute the time in milliseconds
        generate_times.append(start_event_generate.elapsed_time(end_event_generate))

        logits = model.forward(X)

        # calculate loss for only the last_k tokens
        if last_k is not None:
            logits = logits[:, -last_k:, :]
            Y = Y[:, -last_k:]
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), Y.reshape(-1), ignore_index=-1)
        losses[k] = loss.item()

    avg_loss = losses.mean().item()
    avg_generate_time = np.mean(generate_times)
    avg_perplexity = torch.exp(losses).mean().item()

    print('model: ', ckpt_dir, 'dataset: ', dataset)
    print('avg loss: ', avg_loss)
    print('avg perplexity: ', avg_perplexity)
    print('avg generate time: ', avg_generate_time)

if __name__ == '__main__':
    fire.Fire(main)