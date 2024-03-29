from torchio.transforms import RandomMotion, RandomGhosting, RandomBiasField, RandomBlur, RandomNoise, RandomGamma, Compose
from torchio import ScalarImage
from random import randint, uniform
from os import listdir
from tqdm import tqdm

class Augmentation():
    def __init__(self, input_folder, output_folder):
        self.input_folder = f"{input_folder}"
        self.output_folder = output_folder
        self.augmentations = [RandomMotion(degrees=randint(1, 6), translation=randint(1, 6), num_transforms=randint(2, 4)), 
                              RandomGhosting(num_ghosts=randint(4, 10), intensity=randint(0, 3)), 
                              RandomBiasField(coefficients=randint(0, 2), order=randint(1, 3)), 
                              RandomBlur(std=(randint(2, 5), randint(1, 5), randint(2, 5))), 
                              RandomNoise(mean=randint(1, 3), std=1), 
                              RandomGamma(log_gamma=uniform(0, 1))]
    def augment(self):
        selected_augmentations = []
        for i in range(4):
            index = randint(0, len(self.augmentations) - 1) 
            selected_augmentations.append(self.augmentations[index]) 
        transforms = Compose(selected_augmentations)
        for j in tqdm(listdir(self.input_folder)):
            image = ScalarImage(f"{self.input_folder}/{j}/anat/{j}_T1w.nii")
            augmented_image = transforms(image)
            augmented_image.save(f"{self.output_folder}/{j}/anat/{j}_T1w.nii")

if __name__ == "__main__":
    for i in listdir("./data"):
        augs = Augmentation(f"./data/{i}/anat/", f"./data/{i}/anat/")
        augs.augment()