import lj_speech
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from timeit import default_timer
import os
from functools import partial


def normalize(x, mode='linear', audio=True):
    if mode == 'linear':
        _min, _max = x.min(), x.max()
        b = (_min + _max) / (_min - _max)
        a = (1 - b) / _max
        return a * x + b
    elif mode == 'clamp':
        if audio:
            _min, _max = -1, 1
        else:
            _min, _max = 0, 1
        return x.clamp(_min, _max)
    else:
        raise ValueError(f'Unknown mode: {mode}')


def dmap(func, dictionary):
    return {k: func(v) for k, v in dictionary.items()}


def to_device(tensor, device):
    return tensor.to(device)


def inference_routine(item, device):
    return to_device(torch.tensor(item), device)[None, ...]


def inference(model, sample, device='cuda'):
    phonemes, (gt_durations, _) = sample
    phonemes, gt_durations = inference_routine(phonemes, device), inference_routine(gt_durations, device)
    _, predict = model(phonemes, durations=gt_durations)
    predict.squeeze_()

    return predict


def duration_model_forward_step(model, inputs, labels, criterion):
    phonemes, (gt_durations, _) = inputs, labels
    device = phonemes['item'].device
    bs, seq_len = phonemes['item'].shape[:2]
    tok_embedding = model.tok_embedding(phonemes['item'])
    pos_embedding_input = torch.arange(seq_len, device=device).repeat(bs, 1)
    pos_embedding = model.pos_embedding_encoder(pos_embedding_input)
    out = model.encoder(tok_embedding, pos_embedding, pad_mask=~phonemes['mask'])
    durations = model.duration_model(out)

    loss = criterion(durations, gt_durations['item'])
    loss *= gt_durations['mask']

    return loss.mean()


def transformer_forward_step(model, inputs, labels, criterion):
    phonemes, (gt_durations, gt_spectrogram) = inputs, labels
    intermediate_spectrogram, final_spectrogram = model(phonemes['item'], pad_mask=~phonemes['mask'], durations=gt_durations['item'])

    loss = criterion(intermediate_spectrogram, gt_spectrogram['item']) + criterion(final_spectrogram, gt_spectrogram['item'])
    loss *= gt_spectrogram['mask']

    return loss.mean()


def run_epoch(model, dataloader, criterion, optimizer=None, phase='train', mode=None, device='cuda'):
    if mode == 'duration_model':
        forward_step = duration_model_forward_step
    elif mode == 'transformer':
        forward_step = transformer_forward_step
    else:
        raise ValueError(f'Unknown mode: {mode}')

    is_train = (phase == 'train')
    if is_train:
        model.train()
    else:
        model.eval()

    losses = []

    with torch.set_grad_enabled(is_train):
        for phonemes, (gt_durations, gt_spectrogram) in dataloader:
            _to_device = partial(to_device, device=device)
            phonemes, gt_durations, gt_spectrogram = dmap(_to_device, phonemes), dmap(_to_device, gt_durations), dmap(_to_device, gt_spectrogram)
            loss = forward_step(model, phonemes, (gt_durations, gt_spectrogram), criterion)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            losses.append(loss.item())

    return np.mean(losses)


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs,
          logdir, mode=None, vocoder=None, device='cuda', log_freq=10):
    _logdir = os.path.join(logdir, mode)
    os.makedirs(_logdir, exist_ok=True)
    writer = SummaryWriter(_logdir)
    best_loss = 1e6
    for epoch in range(num_epochs):
        t1 = default_timer()

        train_loss = run_epoch(model, train_loader, criterion, optimizer, phase='train', mode=mode, device=device)
        val_loss = run_epoch(model, val_loader, criterion, phase='val', mode=mode, device=device)
        scheduler.step()

        if val_loss < best_loss:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'mapper': model.mapper,
                'token_dict_len': model.token_dict_len,
                'max_seq_len': model.max_seq_len,
                'max_duration': model.max_duration
            }, os.path.join(_logdir, 'model.pth'))
            best_loss = val_loss

        t2 = default_timer()
        epoch_time = t2 - t1

        if epoch % log_freq == 0:
            writer.add_scalar('time', epoch_time, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            if mode == 'transformer' and vocoder is not None:
                for i in range(2):
                    train_spectrogram = inference(model, train_loader.dataset[i], device=device)
                    val_spectrogram = inference(model, val_loader.dataset[i], device=device)
                    for norm_mode in ['linear', 'clamp']:
                        train_audio = normalize(vocoder(train_spectrogram), mode=norm_mode)
                        val_audio = normalize(vocoder(val_spectrogram), mode=norm_mode)
                        writer.add_audio(f'epoch_{epoch}_train_audio_{i}_{norm_mode}', train_audio.detach().cpu(), global_step=epoch)
                        writer.add_audio(f'epoch_{epoch}_val_audio_{i}_{norm_mode}', val_audio.detach().cpu(), global_step=epoch)

                    writer.add_image(f'train_audio_{i}', torch.cat(
                        list(map(normalize, [train_spectrogram.detach().cpu(), torch.tensor(train_loader.dataset[i][1][1])])), dim=1),
                                     global_step=epoch, dataformats='HW')
                    writer.add_image(f'val_audio_{i}', torch.cat(
                        list(map(normalize, [val_spectrogram.detach().cpu(), torch.tensor(val_loader.dataset[i][1][1])])), dim=1),
                                     global_step=epoch, dataformats='HW')


def train_tts(dataset_root, num_epochs_transformer=10, logdir='log_dir', num_epochs_duration=2, device='cuda', num_epochs=None):
    """
    Train the TTS system from scratch on LJ-Speech-aligned stored at
    `dataset_root` for `num_epochs` epochs and save the best model to
    (!!! 'best' in terms of audio quality!) "./TTS.pth".

    dataset_root:
        `pathlib.Path`
        The argument for `lj_speech.get_dataset()`.
    """
    train_loader, val_loader = get_dataloaders(dataset_root)
    train_dataset = train_loader.dataset
    model = Transformer(len(train_dataset.mapper), train_dataset.max_seq_len, train_dataset.max_duration, train_dataset.mapper).to(device)
    vocoder = lj_speech.Vocoder()

    # Train transformer
    criterion_transformer = lambda pred, gt: F.mse_loss(pred, gt, reduction='none') + F.l1_loss(pred, gt, reduction='none')
    optimizer_transformer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler_transformer = CosineAnnealingLR(optimizer_transformer, 10, eta_min=1e-5, last_epoch=-1)
    model.requires_grad_(True)
    model.duration_model.requires_grad_(False)
    train(model, train_loader, val_loader, criterion_transformer, optimizer_transformer, scheduler_transformer,
          num_epochs_transformer, logdir, mode='transformer', vocoder=vocoder, device=device)

    # Train duration model
    criterion_duration = partial(F.mse_loss, reduction='none')
    optimizer_duration = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler_duration = CosineAnnealingLR(optimizer_duration, 10, eta_min=1e-5, last_epoch=-1)
    model.requires_grad_(False)
    model.duration_model.requires_grad_(True)
    train(model, train_loader, val_loader, criterion_duration, optimizer_duration, scheduler_duration,
          num_epochs_duration, logdir, mode='duration_model', device=device)


class TextToSpeechSynthesizer:
    """
    Inference-only interface to the TTS model.
    """
    def __init__(self, checkpoint_path):
        """
        Create the TTS model on GPU, loading its weights from `checkpoint_path`.

        checkpoint_path:
            `str`
        """
        self.vocoder = lj_speech.Vocoder()
        checkpoint = torch.load(checkpoint_path)
        token_dict_len = checkpoint['token_dict_len']
        max_seq_len = checkpoint['max_seq_len']
        max_duration = checkpoint['max_duration']
        self.transformer = Transformer(token_dict_len, max_seq_len, max_duration)
        self.transformer.load_state_dict(checkpoint['state_dict'])
        self.transformer.mapper = checkpoint['mapper']

    def synthesize_from_text(self, text, device='cuda'):
        """
        Synthesize text into voice.

        text:
            `str`

        return:
        audio:
            `torch.Tensor` or `numpy.ndarray`, shape == (1, t)
        """
        phonemes = lj_speech.text_to_phonemes(text)
        return self.synthesize_from_phonemes(phonemes, device=device)

    def synthesize_from_phonemes(self, phonemes, durations=None, device='cuda'):
        """
        Synthesize phonemes into voice.

        phonemes:
            `list` of `str`
            ARPAbet phoneme codes.
        durations:
            `list` of `int`, optional
            Duration in spectrogram frames for each phoneme.
            If given, used for alignment in the model (like during
            training); otherwise, durations are predicted by the duration
            model.

        return:
        audio:
            torch.Tensor or numpy.ndarray, shape == (1, t)
        """
        tokens = ['<bos>', *phonemes, '<eos>']
        if durations is not None:
            durations = [0, *durations, 0]
        indices = list(map(self.transformer.mapper.get, tokens))
        indices = torch.LongTensor(indices).unsqueeze(0).to(device)
        model = self.transformer.to(device)

        with torch.no_grad():
            _, spectrogram = model(indices, durations=durations)

        spectrogram.squeeze_()

        return self.vocoder(spectrogram)


def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'TTS.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'TTS.pth'.
        On Linux (in Colab too), use `$ md5sum TTS.pth`.
        On Windows, use `> CertUtil -hashfile TTS.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'TTS.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here
    md5_checksum = '432f9d117eee9c875dcd0089d47c4ecb'
    google_drive_link = "https://drive.google.com/file/d/1f_6ng4Qsdz1RqYTMwk9UGi2sRSpVoE-3/view?usp=sharing"

    return md5_checksum, google_drive_link


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lj_speech
from functools import partial
from itertools import chain, repeat, islice


class PhonemeDataset(Dataset):
    def __init__(self, lj_speech_dataset, mapper_n_max_stats=None):
        self.lj_speech_dataset = lj_speech_dataset
        if mapper_n_max_stats is None:
            self.construct_mapper()
        else:
            self.mapper, (self.max_seq_len, self.max_duration) = mapper_n_max_stats
        self.text_pad_idx = self.mapper['<pad>']

    def __len__(self):
        return len(self.lj_speech_dataset)

    def __getitem__(self, idx):
        sample = self.lj_speech_dataset[idx]
        phonemes = ['<bos>', *sample['phonemes_code'], '<eos>']
        phonemes = list(map(self.mapper.get, phonemes))

        durations = sample['phonemes_duration']
        durations = [0, *durations, 0]
        spectrogram = sample['spectrogram']

        return phonemes, (durations, spectrogram)

    def construct_mapper(self):
        max_len = 0
        max_duration = 0
        phonemes = set()
        for sample in self.lj_speech_dataset:
            phonemes.update(sample['phonemes_code'])
            cur_len = len(sample['phonemes_code'])
            cur_duration = sample['spectrogram'].shape[1]
            if cur_len > max_len:
                max_len = cur_len
            if cur_duration > max_duration:
                max_duration = cur_duration

        phonemes.update(['<bos>', '<eos>', '<pad>'])
        self.mapper = {phoneme: i for phoneme, i in zip(phonemes, range(len(phonemes)))}
        # take <bos>, <eos> into account
        self.max_seq_len = max_len + 2
        self.max_duration = max_duration


def pad_infinite(iterable, pad):
    return chain(iterable, repeat(pad))


def pad(iterable, size, pad):
    return islice(pad_infinite(iterable, pad), size)


def collate_fn(samples, pad_idx):
    phonemes, labels = zip(*samples)
    durations, spectrogram = zip(*labels)
    max_seq_len = max(map(len, phonemes))
    max_duration = max(map(lambda x: x.shape[1], spectrogram))

    def mask_from_phonemes(phonemes, size):
        mask = torch.zeros(size, dtype=torch.bool)
        mask[:len(phonemes)] = 1
        return mask

    def pad_phonemes(phonemes, size):
        phonemes.extend([pad_idx for _ in range(size - len(phonemes))])
        padded_phonemes = torch.tensor(phonemes, dtype=torch.int64)
        return padded_phonemes

    def mask_from_durations(durations, size):
        mask = torch.zeros(size, dtype=torch.bool)
        # take <bos>, <eos> into account
        mask[1:len(durations)-1] = 1
        return mask

    def pad_durations(durations, size):
        padded_durations = torch.tensor([*durations, *[0 for _ in range(size - len(durations))]], dtype=torch.float32)
        return padded_durations

    def mask_from_spectrogram(spectrogram, size):
        s1, s2 = spectrogram.shape
        mask = torch.zeros(s1, size, dtype=torch.bool)
        mask[:, :s2] = 1
        return mask

    def pad_spectrogram(spectrogram, size):
        _, s = spectrogram.shape
        pad = (0, size - s)
        padded_spectrogram = F.pad(torch.tensor(spectrogram, dtype=torch.float32), pad)
        return padded_spectrogram

    cast2tensor = lambda x: torch.stack(list(x))
    _phonemes, _durations, _spectrogram = {}, {}, {}
    _phonemes['mask'] = cast2tensor(map(partial(mask_from_phonemes, size=max_seq_len), phonemes))
    _phonemes['item'] = cast2tensor(map(partial(pad_phonemes, size=max_seq_len), phonemes))
    _durations['mask'] = cast2tensor(map(partial(mask_from_durations, size=max_seq_len), durations))
    _durations['item'] = cast2tensor(map(partial(pad_durations, size=max_seq_len), durations))
    _spectrogram['mask'] = cast2tensor(map(partial(mask_from_spectrogram, size=max_duration), spectrogram))
    _spectrogram['item'] = cast2tensor(map(partial(pad_spectrogram, size=max_duration), spectrogram))

    return _phonemes, (_durations, _spectrogram)


def get_dataloaders(dataset_root):
    _train_dataset, _val_dataset = lj_speech.get_dataset(dataset_root)
    train_dataset = PhonemeDataset(_train_dataset)
    val_dataset = PhonemeDataset(_val_dataset, (train_dataset.mapper, (train_dataset.max_seq_len, train_dataset.max_duration)))

    _collate_fn = partial(collate_fn, pad_idx=train_dataset.text_pad_idx)
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=2, shuffle=True, collate_fn=_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=2, shuffle=False, collate_fn=_collate_fn)

    return train_loader, val_loader


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
