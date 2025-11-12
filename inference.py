# inference.py
import torch
from vocab import PAD, BOS, EOS
from model import MiniTransformer
from vocab import encode

@torch.no_grad()
def greedy_decode(model, src_sentence, src_vocab, tgt_vocab, tgt_idx2word, device,
                  max_len=20):
    """
    用贪心策略 (greedy search) 生成翻译句子
    """

    # -------- 1️⃣ 编码输入句子 --------
    model.eval()
    src = torch.tensor([encode(src_sentence, src_vocab, max_len)]).to(device)
    src_mask = model.make_src_mask(src)

    # 通过 encoder 得到 memory
    memory = model.positional_encoding(model.src_embedding(src))
    for layer in model.encoder_layers:
        memory = layer(memory, src_mask)

    # -------- 2️⃣ 初始化目标句子 --------
    tgt = torch.tensor([[BOS]], device=device)  # 起始符

    # -------- 3️⃣ 循环生成 --------
    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(tgt)

        out = model.positional_encoding(model.tgt_embedding(tgt))
        for layer in model.decoder_layers:
            out = layer(out, memory, tgt_mask, src_mask)

        logits = model.generator(out)  # [1, len, vocab_size]
        next_token = logits[:, -1, :].argmax(-1).item()  # 取最后一个词的预测

        tgt = torch.cat([tgt, torch.tensor([[next_token]], device=device)], dim=1)

        if next_token == EOS:
            break

    # -------- 4️⃣ 转回文字 --------
    result_tokens = [tgt_idx2word[i] for i in tgt[0].tolist()]
    if EOS in tgt[0]:
        result_tokens = result_tokens[1:-1]  # 去掉 <BOS> 和 <EOS>
    else:
        result_tokens = result_tokens[1:]    # 没生成 EOS 就去掉 BOS

    return " ".join(result_tokens)
