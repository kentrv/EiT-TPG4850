from GAN_3D.Discriminator import Discriminator
from GAN_3D.Generator import Generator
from GAN_3D.LRCN import LRCNModel
from GAN_3D.Train import PhaseOneTrainer, PhaseTwoTrainer, PhaseThreeTrainer
from Pre_processing.shapenet_voxelized import VoxelizedShapeNetDataset
import sys, os

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

input_size =  [4, 1, 32, 32, 32] #Batch_size, channel_size, x, y , z... eg.. [64,1,16,16,16]

generator = Generator(input_size=input_size)

discriminator = Discriminator(input_size=input_size)
lrcn = LRCNModel(dl=32, dh=128)

# Phase 1: Train the GAN
dataset = VoxelizedShapeNetDataset(VOXELIZED_PATH, aligned=True)
chair_dataset = dataset.get_models_in_category("03001627")
phase_one_trainer = PhaseOneTrainer(generator, discriminator, dataset, batch_size=4)
phase_one_trainer.train_generator_only(epochs=20)