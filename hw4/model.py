import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv1d(in_channels, out_channels, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        ]
        if p > 0:
            layers.append(nn.Dropout(p))

        self.conv = nn.Sequential(*layers)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, n_channels, n_head=4, n_hid=1024, p=0.15):
        super(EncoderLayer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(n_channels, n_head, n_hid, p)
        self.conv = ConvBlock(n_channels, n_channels, p)

    def forward(self, x, pad_mask=None):
        seq_len = x.shape[0]
        src_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        out = self.encoder_layer(x, src_mask=src_mask, src_key_padding_mask=pad_mask).permute(1, 2, 0)
        out = self.conv(out).permute(2, 0, 1)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, n_channels, n_head=4, n_hid=1024, p=0.1):
        super(DecoderLayer, self).__init__()
        self.decoder_layer = nn.TransformerEncoderLayer(n_channels, n_head, n_hid, p)
        self.conv = ConvBlock(n_channels, n_channels, p)
        self.n_head = n_head

    def forward(self, x, pad_mask, src_mask=None):
        if src_mask is not None:
            src_mask = src_mask.repeat(self.n_head, 1, 1)
        out = self.conv(x).permute(2, 0, 1)
        out = self.decoder_layer(out, src_mask=src_mask, src_key_padding_mask=pad_mask).permute(1, 2, 0)
        return out


class Encoder(nn.Module):
    def __init__(self, n_channels, n_layers=3):
        super(Encoder, self).__init__()
        self.backbone = nn.Sequential(*[EncoderLayer(n_channels) for _ in range(n_layers)])

    def forward(self, tok_embedding, pos_embedding, pad_mask=None):
        out = (tok_embedding + pos_embedding).transpose(0, 1)
        for layer in self.backbone:
            out = layer(out, pad_mask)

        out = out.transpose(0, 1)

        return out


class Decoder(nn.Module):
    def __init__(self, n_channels, out_channels=80, n_layers=3, n_postprocess=5):
        super(Decoder, self).__init__()
        self.backbone = nn.Sequential(*[DecoderLayer(n_channels) for _ in range(n_layers)])
        self.postprocess = nn.Sequential(*[ConvBlock(n_channels, n_channels, 0) for _ in range(n_postprocess)])
        self.conv_intermediate = nn.Conv1d(n_channels, out_channels, 1)
        self.conv_final = nn.Conv1d(n_channels, out_channels, 1)

    def forward(self, tok_embedding, pos_embedding, durations):
        bs, seq_len = tok_embedding.shape[:2]
        device = tok_embedding.device

        pad_mask = Decoder.construct_pad_mask(bs, seq_len, durations, device)
        # This is may be useless since in duration model bidirectional lstm used, however it trained after rest of model
        src_mask = Decoder.construct_src_mask(bs, seq_len, durations, device)

        out = (tok_embedding + pos_embedding).transpose(1, 2)
        for layer in self.backbone:
            out = layer(out, pad_mask, src_mask=src_mask)

        intermediate = self.conv_intermediate(out)
        out = self.postprocess(out)
        final = self.conv_final(out) + intermediate

        return intermediate, final

    @staticmethod
    def construct_pad_mask(bs, seq_len, durations, device):
        # (seq_len + 1) for sample with maximum duration
        pad_mask = torch.zeros(bs, seq_len + 1, dtype=torch.bool, device=device)
        pad_mask[:, durations.sum(dim=1).type(torch.int64)] = 1
        pad_mask = pad_mask.cumsum(dim=1).type(torch.bool)
        # crop for sample with maximum duration
        pad_mask = pad_mask[:, :-1]

        return pad_mask

    @staticmethod
    def construct_src_mask(bs, seq_len, durations, device):
        # seq_len + 1 because of border cases later crop
        src_mask = torch.zeros(bs, seq_len + 1, seq_len + 1, dtype=torch.int64, device=device)
        # [:, 1:-1] because of <eos>, <bos> symbols
        cumsum = durations[:, 1:-1].cumsum(dim=1).type(torch.int64)
        bs_ids = torch.arange(bs, dtype=torch.int64, device=device).reshape(1, -1).repeat(durations.size(1) - 2, 1).T.flatten()
        src_mask[bs_ids, cumsum.view(-1), cumsum.view(-1)] = 1
        src_mask = src_mask.flip(1).cumsum(1).flip(1)
        src_mask[bs_ids, cumsum.view(-1), cumsum.view(-1)] = 0
        src_mask = src_mask.cumsum(2)[:, :-1, :-1] > 0

        return src_mask


class AlignmentModel(nn.Module):
    def __init__(self):
        super(AlignmentModel, self).__init__()
        self.sigma = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, h, d, size):
        """
        h - tensor [bs, T, d]
        d - tensor [bs, T]
        """
        cumsum = d.cumsum(dim=1).roll(1, 1)
        cumsum[0] = 0
        c = d / 2 + cumsum
        _sigma = 2 * self.sigma ** 2
        t = torch.arange(1, size + 1, dtype=torch.float32, device=h.device) + 0.5
        W = torch.softmax(- (t[None, :, None] - c.unsqueeze(1)) ** 2 / _sigma, dim=-1)
        u = W @ h

        return u


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, in_size, out_size, hid_size, p=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(in_size, hid_size)
        self.fc_2 = nn.Linear(hid_size, out_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class DurationModel(nn.Module):
    def __init__(self, input_size, num_layers=3):
        super(DurationModel, self).__init__()
        assert input_size % 2 == 0, 'input_size should be odd'
        hidden_size = input_size // 2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True)
        self.projection = nn.Sequential(
            PositionwiseFeedforwardLayer(input_size, 1, hidden_size),
            nn.Softplus()
        )

    def forward(self, x):
        out = self.lstm(x.transpose(0, 1))[0].transpose(0, 1)
        out = self.projection(out).squeeze(2)
        return out


class Transformer(nn.Module):
    def __init__(self, token_dict_len, max_seq_len, max_duration, n_channels=256, mapper=None):
        super(Transformer, self).__init__()
        self.encoder = Encoder(n_channels)
        self.decoder = Decoder(n_channels)
        self.alignment_model = AlignmentModel()
        self.duration_model = DurationModel(n_channels)
        self.tok_embedding = nn.Embedding(token_dict_len, n_channels)
        self.pos_embedding_encoder = nn.Embedding(max_seq_len, n_channels)
        self.pos_embedding_decoder = nn.Embedding(max_duration, n_channels)
        self.mapper = mapper
        self.token_dict_len = token_dict_len
        self.max_seq_len = max_seq_len
        self.max_duration = max_duration

    def forward(self, phonemes, pad_mask=None, durations=None):
        device = phonemes.device
        bs, seq_len = phonemes.shape[:2]
        tok_embedding = self.tok_embedding(phonemes)
        pos_embedding_input = torch.arange(seq_len, device=device).repeat(bs, 1)
        pos_embedding = self.pos_embedding_encoder(pos_embedding_input)
        out = self.encoder(tok_embedding, pos_embedding, pad_mask)

        if durations is None:
            durations = self.duration_model(out)

        out = self.alignment_model(out, durations, durations.sum(dim=1).max().type(torch.int32))

        bs, seq_len = out.shape[:2]
        pos_embedding_input = torch.arange(seq_len, device=device).repeat(bs, 1)
        pos_embedding = self.pos_embedding_decoder(pos_embedding_input)
        intermediate_spectrogram, final_spectrogram = self.decoder(out, pos_embedding, durations)

        return intermediate_spectrogram, final_spectrogram
