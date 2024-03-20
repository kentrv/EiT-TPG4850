import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

class VoxelizedShapeNetDataset(Dataset):
    def __init__(self, root_dir, aligned=True, sequence_length=5, use_slices=False):
        self.root_dir = root_dir
        self.aligned = aligned
        self.sequence_length = sequence_length
        self.use_slices = use_slices  # Flag to control slicing
        self.synset_ids = os.listdir(root_dir)
        self.model_paths = []
        for synset_id in self.synset_ids:
            for model_id in os.listdir(os.path.join(root_dir, synset_id)):
                self.model_paths.append((synset_id, model_id, os.path.join(root_dir, synset_id, model_id)))

    def __len__(self):
        return len(self.model_paths)

    def __getitem__(self, idx):
        synset_id, model_id, model_path = self.model_paths[idx]
        if self.aligned:
            voxel_array = np.load(os.path.join(model_path, 'aligned_voxel.npy'))
        else:
            voxel_array = np.load(os.path.join(model_path, 'voxel.npy'))
        if self.use_slices:
            voxel_array = self.preprocess_volume(voxel_array, self.sequence_length)
        return synset_id, model_id, torch.from_numpy(voxel_array)

    def preprocess_volume(self, volume, sequence_length):
        dl, dh, dw = volume.shape
        padding = (sequence_length - 1) // 2
        padded_volume = np.pad(volume, ((padding, padding), (0, 0), (0, 0)), mode='constant', constant_values=0)
        num_sequences = dl
        sequences = np.zeros((num_sequences, sequence_length, dh, dw))
        for i in range(num_sequences):
            start_idx = i
            end_idx = start_idx + sequence_length
            sequences[i] = padded_volume[start_idx:end_idx, :, :]
        return sequences

    def get_models_in_category(self, target_synset_id):
        indices = [i for i, (synset_id, _, _) in enumerate(self.model_paths) if synset_id == target_synset_id]
        return Subset(self, indices)
    
    
    def enable_slices(self, use_slices=True):
        self.use_slices = use_slices