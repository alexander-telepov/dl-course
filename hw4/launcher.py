from pathlib import Path
DATASET_ROOT = Path("./LJ-Speech-aligned")
from part2_solution import train_tts
# train_tts(DATASET_ROOT, 100, num_epochs_duration=20, logdir='debug_2/add_decoder_mask')
# train_tts(DATASET_ROOT, 100, num_epochs_duration=20, logdir='debug_2/decoder_mask_off')
# train_tts(DATASET_ROOT, 100, num_epochs_duration=20, logdir='debug_2/encoder_masks_off')
# train_tts(DATASET_ROOT, 2, num_epochs_duration=2, logdir='debug_2/test_bug', device='cpu')
# train_tts(DATASET_ROOT, 2, num_epochs_duration=2, logdir='debug_2/test_train')
train_tts(DATASET_ROOT, 10, num_epochs_duration=10, logdir='debug_2/final')