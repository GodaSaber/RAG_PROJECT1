import torch
import torch.nn as nn
import json
import os
import re


# =========================
# Arabic + English Tokenizer (بسيط من صنعك)
# =========================
class SimpleTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build_vocab(self, texts):
        word_freq = {}
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Keep top words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words[: self.vocab_size - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text, max_len=50):
        words = self._tokenize(text)
        indices = []
        for word in words[:max_len]:
            indices.append(self.word2idx.get(word, 1))  # 1 = UNK

        # Padding
        while len(indices) < max_len:
            indices.append(0)  # 0 = PAD

        return indices

    def _tokenize(self, text):
        text = text.lower().strip()
        # Split Arabic and English words
        words = re.findall(r"[\u0600-\u06FF]+|[a-zA-Z]+|[0-9]+", text)
        return words

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
            self.idx2word = {v: k for k, v in self.word2idx.items()}


# =========================
# Neural Network Model
# =========================
class QuestionClassifier(nn.Module):
    def __init__(
        self, vocab_size=5000, embed_dim=64, hidden_dim=128, num_classes=3, max_len=50
    ):
        super(QuestionClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, max_len)
        x = self.embedding(x)  # (batch, max_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, max_len)

        x = self.relu(self.conv1(x))  # (batch, hidden_dim, max_len)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))  # (batch, hidden_dim, max_len)

        x = self.pool(x)  # (batch, hidden_dim, 1)
        x = x.squeeze(2)  # (batch, hidden_dim)

        x = self.relu(self.fc1(x))  # (batch, 64)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, num_classes)

        return x


# =========================
# Category Labels
# =========================
CATEGORIES = {0: "educational", 1: "legal", 2: "general"}  # تعليمي  # قانوني  # عام

CATEGORIES_AR = {0: "تعليمي", 1: "قانوني", 2: "عام"}
