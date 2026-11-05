"""Extended Hamming SECDED: single-bit correction, double-bit rejection.

This protects address code bits, never semantic fact selection.
"""
import math
import torch


class SECDED:
    def __init__(self, classes):
        self.classes = classes
        self.k = max(1, math.ceil(math.log2(classes)))
        self.r = 2
        while 2 ** self.r < self.k + self.r + 1:
            self.r += 1
        self.n = self.k + self.r
        self.data = [p for p in range(1, self.n + 1) if p & (p - 1)]
        self.width = self.n + 1

    def encode(self, payload):
        code = torch.zeros((*payload.shape, self.width), dtype=torch.long, device=payload.device)
        for i, p in enumerate(self.data):
            code[..., p - 1] = (payload >> i) & 1
        for i in range(self.r):
            p = 1 << i
            positions = [j - 1 for j in range(1, self.n + 1) if j & p]
            code[..., p - 1] = code[..., positions].sum(-1) % 2
        code[..., -1] = code[..., :-1].sum(-1) % 2
        return code

    def decode(self, code):
        code = code.long().clone()
        syndrome = torch.zeros(code.shape[:-1], dtype=torch.long, device=code.device)
        for i in range(self.r):
            positions = [j - 1 for j in range(1, self.n + 1) if j & (1 << i)]
            syndrome += (code[..., positions].sum(-1) % 2) * (1 << i)
        odd = code.sum(-1) % 2 == 1
        invalid = ((syndrome != 0) & ~odd) | (syndrome > self.n)
        corrected = odd & ~invalid
        pos = torch.where(syndrome == 0, self.n, syndrome - 1).clamp(0, self.n)
        correction = torch.zeros_like(code).scatter_(-1, pos.unsqueeze(-1), corrected.long().unsqueeze(-1))
        code ^= correction
        payload = sum(code[..., p - 1] * (1 << i) for i, p in enumerate(self.data))
        invalid |= payload >= self.classes
        payload = torch.where(invalid, -torch.ones_like(payload), payload)
        return payload, corrected, invalid

    def experiment(self):
        payload = torch.arange(self.classes)
        clean = self.encode(payload)
        single, double, total_single, total_double = 0, 0, 0, 0
        for i in range(self.width):
            x = clean.clone()
            x[:, i] ^= 1
            single += int((self.decode(x)[0] == payload).sum())
            total_single += self.classes
            for j in range(i + 1, self.width):
                y = x.clone()
                y[:, j] ^= 1
                double += int(self.decode(y)[2].sum())
                total_double += self.classes
        return dict(single_bit_recovery=single / total_single, double_bit_rejection=double / total_double,
                    single_cases=total_single, double_cases=total_double,
                    caveat="Bit corruption only. A valid code for a wrong fact is not corrected.")
