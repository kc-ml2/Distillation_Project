import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import eval_validation_tool as evt


class FakeTokenizerOutput:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        return self


class FakeTokenizer:
    enc_token_id = 999

    def __call__(self, texts, padding="max_length", truncation=True, max_length=35, return_tensors="pt"):
        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        for i, t in enumerate(texts):
            ids[i, 0] = int(t.split("_")[-1])
        mask = torch.ones(len(texts), max_length, dtype=torch.long)
        return FakeTokenizerOutput(ids, mask)


class FakeTextEncoderOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeTextEncoder(nn.Module):
    """Embeds token-id-at-position-0 as a one-hot vector. enc_token_id (999)
    is out of range so rerank calls (which overwrite position 0 with
    enc_token_id) deterministically produce an all-zero hidden state."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, input_ids, attention_mask=None, mode=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                return_dict=True):
        batch, seq_len = input_ids.shape
        hidden = torch.zeros(batch, seq_len, self.embed_dim)
        for b in range(batch):
            tok = int(input_ids[b, 0].item())
            if 0 <= tok < self.embed_dim:
                hidden[b, 0, tok] = 1.0
        return FakeTextEncoderOutput(hidden)


class FakeVisualEncoder(nn.Module):
    def forward(self, image):
        return image.unsqueeze(1)  # [B, embed_dim] -> [B, 1, embed_dim]


class CountingLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        self.call_count += 1
        return super().forward(x)


class FakeRetrievalModel(nn.Module):
    def __init__(self, embed_dim=4):
        super().__init__()
        self.tokenizer = FakeTokenizer()
        self.text_encoder = FakeTextEncoder(embed_dim)
        self.text_proj = nn.Identity()
        self.visual_encoder = FakeVisualEncoder()
        self.vision_proj = nn.Identity()
        self.itm_head = CountingLinear(embed_dim, 2)


class FakeRetrievalDataset(Dataset):
    """n images, n captions, 1:1 matched by index — image i's embedding and
    caption i's embedding are constructed to be the same one-hot vector, so
    a correct implementation always ranks the true match first."""

    def __init__(self, n=4):
        self.text = [f"item_{i}" for i in range(n)]
        self.image = list(range(n))
        self.txt2img = {i: i for i in range(n)}
        self.img2txt = {i: [i] for i in range(n)}
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        vec = torch.zeros(self.n)
        vec[index] = 1.0
        return vec, index


class EvaluateRetrievalSplitTest(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.model = FakeRetrievalModel(embed_dim=self.n)
        self.dataset = FakeRetrievalDataset(n=self.n)
        self.loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        self.device = torch.device("cpu")
        self.config = {"k_test": 2}

    def test_itc_only_skips_itm_head_and_gets_perfect_recall(self):
        scores_i2t, scores_t2i = evt.evaluate_retrieval_itc(
            self.model, self.loader, self.device, self.config)

        self.assertEqual(self.model.itm_head.call_count, 0)
        self.assertEqual(scores_i2t.shape, (self.n, self.n))
        self.assertEqual(scores_t2i.shape, (self.n, self.n))

        metrics = evt.itm_eval(scores_i2t, scores_t2i,
                                self.dataset.txt2img, self.dataset.img2txt)
        self.assertEqual(metrics["r_mean"], 100.0)

    def test_itm_rerank_invokes_itm_head_and_gets_perfect_recall(self):
        scores_i2t, scores_t2i = evt.evaluate_retrieval_itm(
            self.model, self.loader, self.device, self.config)

        self.assertGreater(self.model.itm_head.call_count, 0)
        self.assertEqual(scores_i2t.shape, (self.n, self.n))
        self.assertEqual(scores_t2i.shape, (self.n, self.n))

        metrics = evt.itm_eval(scores_i2t, scores_t2i,
                                self.dataset.txt2img, self.dataset.img2txt)
        self.assertEqual(metrics["r_mean"], 100.0)


if __name__ == "__main__":
    unittest.main()
