import torch

data = torch.load('datasets/eeg_5_95_std.pth', map_location='cpu')

# Check dataset structure
print('=== Data Structure ===')
if 'dataset' in data and len(data['dataset']) > 0:
    sample = data['dataset'][0]
    print(f'Dataset item keys: {sample.keys()}')
    for k, v in sample.items():
        if k != 'eeg':
            print(f'  {k}: {v} (type: {type(v).__name__})')

print('\n=== Labels (WordNet IDs) ===')
labels = data['labels']
print(f'Total labels: {len(labels)}')
print(f'Sample: {labels[:5]}')

print('\n=== Images ===')
images = data['images']
print(f'Total images: {len(images)}')
# Show some image name mappings
count = 0
for k, v in images.items():
    if count < 5:
        print(f'  Index {k} -> {v}')
        count += 1
