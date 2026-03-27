from torch_compat import load_full
import numpy as np

data = load_full('../datasets/eeg_5_95_std.pth', map_location='cpu')
print(f'Total samples: {len(data["dataset"])}')

splits = load_full('../datasets/block_splits_by_image_all.pth', map_location='cpu')
print(f'Train split size: {len(splits["splits"][0]["train"])}')
print(f'Test split size: {len(splits["splits"][0]["test"])}')

# Check EEG length distribution
lengths = []
for i, sample in enumerate(data['dataset']):
    eeg = sample.get('eeg', None)
    if eeg is not None:
        try:
            tlen = int(eeg.size(1))
        except:
            tlen = int(eeg.shape[1])
        lengths.append(tlen)

print(f'EEG length range: {min(lengths)} - {max(lengths)}')
print(f'EEG length in 450-600: {sum(1 for l in lengths if 450 <= l <= 600)} / {len(lengths)}')

# Check by split
for split_name in ['train', 'test']:
    split_indices = splits["splits"][0][split_name]
    valid_count = 0
    for ds_idx in split_indices:
        if ds_idx < 0 or ds_idx >= len(data['dataset']):
            continue
        sample = data['dataset'][ds_idx]
        eeg = sample.get('eeg', None)
        if eeg is None:
            continue
        try:
            tlen = int(eeg.size(1))
        except:
            tlen = int(eeg.shape[1])
        if 450 <= tlen <= 600:
            valid_count += 1
    print(f'{split_name}: {valid_count} samples with EEG length 450-600 (out of {len(split_indices)} in split)')
