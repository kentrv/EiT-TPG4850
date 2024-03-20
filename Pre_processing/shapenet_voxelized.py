import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

class VoxelizedShapeNetDataset(Dataset):
    def __init__(self, root_dir, aligned=True):
        self.root_dir = root_dir
        self.aligned = aligned
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
        return synset_id, model_id, torch.from_numpy(voxel_array)
    
    def get_models_in_category(self, target_synset_id):
        indicies = [i for i, (synset_id, _, _) in enumerate(self.model_paths) if synset_id == target_synset_id]
        return Subset(self, indicies)