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
