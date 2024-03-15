import torchio as tio
import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from dipy.segment.mask import median_otsu
from pyrobex import robex
from tqdm import tqdm
import torch

# test for github push

class Preprocessing:
    """A class for preprocessing MRI scans.

    Attributes:
        input_folder (str): The folder containing the scans to preprocess.
        output_folder (str): The folder where preprocessed scans will be stored.
    """
    def __init__(self, input_folder, output_folder):
        """
        Initializes the Preprocessing class with input and output folders.

        Args:
            input_folder (str): The folder containing the scans to preprocess.
            output_folder (str): The folder where preprocessed scans will be stored.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        

    def run_preprocess(self):
        """Runs the preprocessing steps for all scans in the input folder."""
        for i in tqdm(self.files):
            self.correctBias(input_image_path=f"./{self.input_folder}/{i}")
            self.extractBrain(i)
            print("successful!\n")
        print("Preprocessing Complete!")

    def correctBias(self, input_image_path=None, image_created = False, image = None, image_mask = None):
        """
        Corrects bias in the MRI scan.

        Args:
            input_image_path (str): Path to the input MRI image.
            image_created (bool): Indicates if the image has already been created.
            image: The image object.
            image_mask: The mask image object.

        Returns:
            sitk.Image: The bias-corrected MRI image.
        """
        input_image = image

        if image_created == False:
            input_image = sitk.ReadImage(input_image_path, sitk.sitkFloat32)
            self.createMRIMask(input_image_path)
        
        mask_image = sitk.ReadImage("temp_mask.nii", sitk.sitkUInt8)
        os.remove("temp_mask.nii")


        shrinkFactor = 3

        shrunk_image = sitk.Shrink(
            input_image, [shrinkFactor] * input_image.GetDimension()
        )
        shrunk_mask = sitk.Shrink(
            mask_image, [shrinkFactor] * input_image.GetDimension()
        )

        numFittingLevels = 4

        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        corrected_image = corrector.Execute(shrunk_image, shrunk_mask)

        log_bias_field = corrector.GetLogBiasFieldAsImage(input_image)

        corrected_image_full_resolution = input_image / \
            sitk.Exp(log_bias_field)

        if image_created == False:
            sitk.WriteImage(corrected_image_full_resolution, "bixed.nii")
        
        elif image_created == True:
            return corrected_image_full_resolution
            print("Bias corrected!")

    def createMRIMask(self, input_image_path, image_created = False):
        """
        Creates a mask for the MRI scan.

        Args:
            input_image_path (str): Path to the input MRI image.
            image_created (bool): Indicates if the image has already been created.

        Returns:
            None
        """
        img = nib.load(input_image_path)
        data = np.squeeze(img.get_fdata())
        b0_mask, mask = median_otsu(data, median_radius=2, numpass=1)
        mask_img = nib.Nifti1Image(mask.astype(np.float32), img.affine)
        nib.save(mask_img, "temp_mask.nii")
        print("Mask created!")

    def extractBrain(self, input_image_path):
        """
        Performms brain extraction on the MRI scan.

        Args:
            input_image_path (str): Path to the input MRI image.

        Returns:
            None
        """
        image = nib.load("./bixed.nii")
        stripped = robex.robex_stripped(image)
        output_path = input_image_path[:-4]
        nib.save(stripped, f"./{self.output_folder}/{output_path}_preprocessed.nii")
        os.remove("./bixed.nii")

    def minMaxNormalization(self, image):
        """
        Performs min-max normalization on the image.

        Args:
            image: The input image.

        Returns:
            torch.Tensor: The normalized image.
        """
        tensor_min = torch.min(image)
        tensor_max = torch.max(image)
        normalized_image = (image - tensor_min) / (tensor_max - tensor_min)
        return normalized_image

class DatasetPreparation(Preprocessing):
    """
    Prepares training and testing datasets using methods from Preprocessing class

    Attributes:
        input_folder_path (str): The folder containing the scans to preprocess.
        training_folder_path (str, optional): The folder that will store training scans. Defaults to None.
        ideal_folder_path (str, optional): The folder that will store ideal or ground truth scans. Defaults to None.
    """
    def __init__(self, input_folder, output_folder) -> None:
        """
        Initializes DatasetPreparation object.

        Args:
            input_folder_path (str): The path to the input dataset folder.
            training_folder_path (str, optional): The path to the folder that stores the training scans. Defaults to None.
            ideal_folder_path (str, optional): The path to the folder that will store the ideal scans. Defaults to None.
        """
        Preprocessing.__init__(self, input_folder, output_folder)
        self.input_files = os.listdir(input_folder)
    
    def baselineImprov(self):
        """
        Creates ideal data by correcting bias fields, conducting min-max normalization, and removing noise. 
        """
        denoise = sitk.MinMaxCurvatureFlowImageFilter()
        for i in tqdm(self.input_files):
            image_path = f"{self.input_folder}/{i}/anat/{i}_T1w.nii"
            image = sitk.ReadImage(image_path, sitk.sitkFloat32)
            image_mask = super().createMRIMask(image_path, image_created=True)
            bias_corrected_image = super().correctBias(input_image_path=None, image_created=True, image=image, image_mask=image_mask)
            
            denoised_image = denoise.Execute(bias_corrected_image)
            print("Image denoised!")
            
            image_array = sitk.GetArrayFromImage(denoised_image)
            image_tensor = torch.tensor(image_array, dtype=torch.float32)
            normalized_image = super().minMaxNormalization(image_tensor)
            print("Image normalized!")
            
            norm_img_array = np.array(normalized_image)
            norm_img_nifti = nib.Nifti1Image(norm_img_array, affine=np.eye(4))
            nib.save(norm_img_nifti, f"{self.output_folder}/{i}_T1w.nii")
            print("Enhancement Successful!")

if __name__ == "__main__":
    directory_1 = "/home/ramkumars@acct.upmchs.net/Projects/3D-Harmonization/data/scanner_1/anat"
    directory_1_out = "/home/ramkumars@acct.upmchs.net/Projects/3D-Harmonization/data/scanner_1_processed"
    directory_2 = "/home/ramkumars@acct.upmchs.net/Projects/3D-Harmonization/data/scanner_2/anat"
    directory_2_out = "/home/ramkumars@acct.upmchs.net/Projects/3D-Harmonization/data/scanner_2_processed"

    dp = DatasetPreparation(directory_1, directory_1_out)
    dp.baselineImprov()
    dp = DatasetPreparation(directory_2, directory_2_out)
    dp.baselineImprov()

