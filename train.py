# train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from vocab import build_vocab, encode, PAD
from toy_data import src_sentences, tgt_sentences
from model import MiniTransformer

# ---------- 1️⃣ 数据集定义 ----------
class ToyDataset(Dataset):
    def __init__(self, src_sents, tgt_sents, src_vocab, tgt_vocab, max_len=10):
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, idx):
        src = encode(self.src_sents[idx], self.src_vocab, self.max_len)
        tgt = encode(self.tgt_sents[idx], self.tgt_vocab, self.max_len)
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    return torch.stack(srcs), torch.stack(tgts)

# ---------- 2️⃣ 训练函数 ----------
def train():
    # 1. 构建词表
    src_vocab, _ = build_vocab(src_sentences)
    tgt_vocab, tgt_idx2word = build_vocab(tgt_sentences)

    # 2. 构建数据集与 DataLoader
    dataset = ToyDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=8)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 3. 初始化模型、损失函数、优化器
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MiniTransformer(len(src_vocab), len(tgt_vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 4. 训练循环
    for epoch in range(30):
        model.train()
        total_loss = 0

        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)

            # 拆分 decoder 输入和标签
            tgt_input = tgt[:, :-1]  # 去掉最后一个词
            tgt_output = tgt[:, 1:]  # 去掉第一个词

            # 前向传播
            logits = model(src, tgt_input)  # [B, tgt_len-1, vocab_size]

            # 计算损失
            logits = logits.reshape(-1, logits.size(-1))
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(logits, tgt_output)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(loader):.4f}")

    print("Training complete ✅")
    return model, src_vocab, tgt_vocab, tgt_idx2word, device


if __name__ == "__main__":
    train()
