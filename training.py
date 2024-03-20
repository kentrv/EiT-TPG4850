from GAN_3D.Discriminator import Discriminator
from GAN_3D.Generator import Generator
from GAN_3D.LRCN import LRCNModel
from GAN_3D.Train import PhaseOneTrainer, PhaseTwoTrainer, PhaseThreeTrainer
from Pre_processing.shapenet_voxelized import VoxelizedShapeNetDataset
import sys, os
import numpy as np
import copy as Copy

# Initialize ShapeNetCore dataset
if sys.platform == "win32":
    SHAPENET_PATH = "D:/ShapeNet"
    # Define the root directory for the voxelized models
    VOXELIZED_PATH = "D:/ShapeNet_Voxelized"
    
elif sys.platform == "linux":
    user = os.environ.get('USER')
    SHAPENET_PATH = "/media/"+user+"/Elements/ShapeNet"
    # Define the root directory for the voxelized models
    VOXELIZED_PATH = "/media/"+user+"/Elements/ShapeNet_Voxelized"
else:
    print("Unsupported OS. Exiting...")
    sys.exit()

        
def postprocess_volume(volume, sequence_length=5):
    """
    Postprocess the volume by combining the sequences of slices into a single volume.
    
    Args:
    volume: A numpy array of shape [dl, sequence_length, dh, dh], that is: a numpy array representing dl sequences of length sequence_length containing a dl x dl 2D voxel array.
    sequence_length: How many slices in the sequence per slice.
    
    Returns: A numpy array of shape (dh, dh, dh) representing the voxel array.
    """
    dl, sequence_length, dh, _ = volume.shape
    final_volume = np.zeros((dh, dh, dh))
    
    # Iterate over each sequence
    for i in range(dl):
        for j in range(sequence_length):
            # Assuming each sequence directly maps to a section of the final volume
            # This logic may need adjustment based on the actual structure of your data
            if i*sequence_length + j < dh:
                final_volume[:, :, i*sequence_length + j] += volume[i, j, :, :]

    return final_volume

input_size =  [4, 1, 32, 32, 32] #Batch_size, channel_size, x, y , z... eg.. [64,1,16,16,16]

generator = Generator(input_size=input_size)

discriminator = Discriminator(input_size=input_size)
lrcn = LRCNModel(dl=32, dh=128)

# Phase 1: Train the GAN
dataset = VoxelizedShapeNetDataset(VOXELIZED_PATH, aligned=True)
chair_dataset = dataset.get_models_in_category("03001627")
chair_dataset = Copy.deepcopy(chair_dataset)
#phase_one_trainer = PhaseOneTrainer(generator, discriminator, chair_dataset, batch_size=4)
#phase_one_trainer.train_generator_only(epochs=20)
#phase_one_trainer.train_jointly(epochs=20)

dataset.enable_slices(use_slices=True)

sliced_data = dataset.get_models_in_category("03001627")
phase_two_trainer = PhaseTwoTrainer(lrcn, chair_dataset, sliced_data, batch_size=4)
phase_two_trainer.train(epochs=20)