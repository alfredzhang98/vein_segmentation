import os
import sys
import math
import random
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
from config import DataInfo
from albumentations.core.transforms_interface import DualTransform


class AugmentedDataset(DataInfo):
    def __init__(self):
        super().__init__()
        self.target_size = (self.max_height, self.max_width)  # (height, width) - optimized for ultrasound images
        np.random.seed(self.seed)
        random.seed(self.seed)

        # pipeline for data augmentation
        self.transform = A.Compose([
            # upscale the image to 1.5x size
            A.Resize(int(self.target_size[0]*1.5), int(self.target_size[1]*1.5), 
             interpolation=cv2.INTER_CUBIC, p=1.0),
            A.Rotate(limit=10, p=0.5,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_REPLICATE, fill=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # resize back to target size
            A.Resize(self.target_size[0], self.target_size[1], 
            interpolation=cv2.INTER_AREA, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.6
            ),
            A.GaussNoise(std_range=(0.05, 0.1), p=0.3),
            # A.RandomScale(scale_limit=(-0.75, 0.0), p=0.1),
            # Black padding
            # A.PadIfNeeded(
            # min_height=self.target_size[0],
            # min_width=self.target_size[1],
            # border_mode=cv2.BORDER_REPLICATE,
            # p=1.0
            # ),
        ], additional_targets={'mask': 'mask'})

        # pipeline for smart padding
        # self.smart_pad_transform = UltrasoundPatchPad(
        #     target_size=self.target_size,
        #     patch_size=32
        # )

        # final transformation to PyTorch Tensor
        self.final_transform = A.Compose([
            # for image it is (-1, 1) normalization, for mask it is (0, 1)
            A.Normalize(mean=[0.5,], std=[0.5,], p=1.0),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    
    def load_image_and_mask(self, image_path: Path, mask_path: Path) -> tuple:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"The image file does not exist: {image_path}")
        if not Path(mask_path).exists():
            raise FileNotFoundError(f"The mask file does not exist: {mask_path}")

        # load the grayscale image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image file: {image_path}")

        # keep single-channel grayscale image
        # load the segmentation mask - binary mask for vessel segmentation
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask file: {mask_path}")

        # Key processing: mask value remapping (255->1, 0->0)
        # This ensures compatibility with PyTorch BCE loss function
        mask = self.__remap_mask(mask, self.mask_class_map)
        
        return image, mask

    def __remap_mask(self, mask: np.ndarray, mapping: dict) -> np.ndarray:
        result = np.zeros_like(mask, dtype=np.uint8)
        for k, v in mapping.items():
            result[mask == k] = v
        return result
    
    @staticmethod
    def crop_image(img: np.ndarray,
                   crop_top: int = 0,
                   crop_bottom: int = 0,
                   crop_left: int = 0,
                   crop_right: int = 0) -> np.ndarray:
        """
        Crop the given image by removing specified pixels from each border.

        Parameters
        ----------
        img : np.ndarray
            The input image array. Can be grayscale (H x W) or color (H x W x C).
        crop_top : int, optional
            Number of pixels to crop from the top border. Default is 100.
        crop_bottom : int, optional
            Number of pixels to crop from the bottom border. Default is 100.
        crop_left : int, optional
            Number of pixels to crop from the left border. Default is 100.
        crop_right : int, optional
            Number of pixels to crop from the right border. Default is 100.

        Returns
        -------
        np.ndarray
            The cropped image.

        Raises
        ------
        TypeError
            If any crop parameter is not an integer.
        ValueError
            If any crop parameter is negative or too large for the image dimensions.
        """
        # Validate parameters
        for name, value in (('crop_top', crop_top),
                            ('crop_bottom', crop_bottom),
                            ('crop_left', crop_left),
                            ('crop_right', crop_right)):
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {type(value)}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        # Get image dimensions
        if img.ndim == 2:
            h, w = img.shape
        elif img.ndim == 3:
            h, w, _ = img.shape
        else:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}")

        # Ensure cropping isn't too large
        if crop_top + crop_bottom >= h:
            raise ValueError(f"crop_top + crop_bottom ({crop_top + crop_bottom}) "
                             f"is too large for image height ({h})")
        if crop_left + crop_right >= w:
            raise ValueError(f"crop_left + crop_right ({crop_left + crop_right}) "
                             f"is too large for image width ({w})")

        # Perform crop
        return img[crop_top:h - crop_bottom,
                   crop_left:w - crop_right]

    def augment_single_image(self, image : np.ndarray, mask: np.ndarray, num_augmentations: int) -> tuple:
        augmented_images = []
        augmented_masks = []
        
        for i in range(num_augmentations):
            try:
                # Use the SAME cropped image for each augmentation iteration
                # apply random data augmentation transformations
                augmented = self.transform(image=image, mask=mask)
                aug_image = augmented['image']
                aug_mask = augmented['mask']

                assert aug_image.shape[:2] == aug_mask.shape, "Enhanced image and mask dimensions do not match"
                assert aug_image.shape[:2] == self.target_size, f"Error size: {aug_image.shape[:2]} vs {self.target_size}"

                # final transformation: normalization + convert to PyTorch Tensor
                final_transformed = self.final_transform(image=aug_image, mask=aug_mask)
                
                # Extract tensor data (convert to numpy for batch processing)
                tensor_image = final_transformed['image'].numpy()  # (C, H, W) - 单通道
                tensor_mask = final_transformed['mask'].numpy()    # (H, W)
                
                # Make sure image is single-channel format (1, H, W)
                if tensor_image.ndim == 2:  # If shape is (H, W)
                    tensor_image = tensor_image[np.newaxis, ...]  # Add channel dimension -> (1, H, W)
                elif tensor_image.ndim == 3 and tensor_image.shape[0] != 1:
                    raise ValueError(f"Image should be single-channel, got shape: {tensor_image.shape}")

                # Make sure mask is single-channel format (1, H, W)
                if tensor_mask.ndim == 2:  # If shape is (H, W)
                    tensor_mask = tensor_mask[np.newaxis, ...]  # Add channel dimension -> (1, H, W)
                elif tensor_mask.ndim == 3 and tensor_mask.shape[0] != 1:
                    raise ValueError(f"Mask should be single-channel, got shape: {tensor_mask.shape}")

                # Data type and range validation
                assert tensor_image.dtype == np.float32, f"Image data type error: {tensor_image.dtype}"
                assert tensor_mask.dtype in [np.uint8, np.float32], f"Mask data type error: {tensor_mask.dtype}"
                assert tensor_mask.max() <= 1, f"Mask value out of range: {tensor_mask.max()}"
                assert tensor_image.shape[0] == 1, f"Image should be single-channel: {tensor_image.shape}"
                assert tensor_mask.shape[0] == 1, f"Mask should be single-channel: {tensor_mask.shape}"

                augmented_images.append(tensor_image)
                augmented_masks.append(tensor_mask)
                
            except Exception as e:
                raise RuntimeError(f"Data augmentation failed for image {i+1}: {e}")
        
        if len(augmented_images) == 0:
            raise RuntimeError("All augmentation attempts failed for this image.")

        return augmented_images, augmented_masks

    def generate_augmented_data(self) -> tuple:
        # read metadata file
        if not self.meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_file}")

        df = pd.read_csv(self.meta_file)
        print(f"Loaded metadata with {len(df)} samples.")

        # Count original data by class
        class_counts = df['mask_status'].value_counts()
        print(f"Original data distribution:")
        for class_name, count in class_counts.items():
            target_aug = self.target_aug_times.get(class_name, 0)
            print(f"  {class_name}: {count} -> {count * target_aug} after augmentation")

        # Store augmentation results
        all_images = []
        all_masks = []
        all_image_type = []
        all_filenames = []

        # Statistics
        success_count = 0
        error_count = 0
        skip_count = 0 
        class_generated = {cls: 0 for cls in self.target_aug_times.keys()}

        # Process each sample row by row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples", disable=False,
                            bar_format='{desc}: {n}/{total} [{elapsed}]'):
            try:
                image_name = row['filename']
                mask_status = row['mask_status']
                # Validate class existence and skip unnecessary augmentations (e.g., "pass")
                if mask_status not in self.target_aug_times:
                    skip_count += 1
                    continue

                # Check if mask_path is a valid string (avoid NaN, etc.)
                if pd.isna(row['mask_path']) or not isinstance(row['mask_path'], str):
                    skip_count += 1
                    continue

                # Parse file paths (handle Windows path separators)
                if sys.platform.startswith("linux"):
                    image_path = Path(row['relative_path'].replace('\\', '/'))
                    mask_path = Path(row['mask_path'].replace('\\', '/'))
                else:
                    image_path = Path(row['relative_path'])
                    mask_path = Path(row['mask_path'])
                
                # Make sure the augmentation times for different mask_status
                aug_times = self.target_aug_times[mask_status]

                # If augmentation times is 0, skip this sample
                if aug_times <= 0:
                    skip_count += 1
                    continue
                
                # load image and mask
                image, mask = self.load_image_and_mask(image_path, mask_path)

                # Adjust crop parameters to be more conservative
                # Original image is typically 557x528, target is 576x544
                # Use smaller crop values to avoid very small intermediate images
                cropped_image = self.crop_image(image,
                                            crop_top=10,
                                            crop_bottom=15,
                                            crop_left=50,
                                            crop_right=50)
                cropped_mask = self.crop_image(mask,
                                            crop_top=10,
                                            crop_bottom=15,
                                            crop_left=50,
                                            crop_right=50)
        

                # perform data augmentation
                aug_images, aug_masks = self.augment_single_image(cropped_image, cropped_mask, aug_times)

                # Add to the total dataset
                all_images.extend(aug_images)
                all_masks.extend(aug_masks)
                all_image_type.extend([mask_status] * len(aug_images))
                image_name = image_name.split('.')[0]  # Remove file extension for consistency
                all_filenames.extend(f"{image_name}_{i:04d}" for i in range(len(aug_images)))

                # Update statistics
                success_count += 1
                class_generated[mask_status] += len(aug_images)
                
            except Exception as e:
                error_count += 1
                raise RuntimeError(f"Error processing sample {idx+1}: {e}")
        
        print(f"Data augmentation complete: {len(all_images)} samples generated")
        print(f"Class-wise generation:")
        for class_name, generated_count in class_generated.items():
            original_count = class_counts.get(class_name, 0)
            expected_count = original_count * self.target_aug_times[class_name]
            if expected_count > 0:
                percentage = generated_count/expected_count*100
                print(f"  {class_name}: {generated_count}/{expected_count} ({percentage:.1f}%)")
            else:
                print(f"  {class_name}: {generated_count}/0 (no original data or augmentation disabled)")
        
        if len(all_images) == 0:
            raise RuntimeError("No augmented data was successfully generated!")

        return all_images, all_masks, all_image_type, all_filenames

    def save_augmented_data(self, images: list, masks: list, image_type: list, filenames: list, save_png: bool = True) -> None:
        """
        Save augmented data to NPZ compressed file, with optional PNG image file saving
        
        Args:
            images: List of augmented images (each with shape CxHxW)
            masks: List of augmented masks (each with shape HxW, converted to CxHxW where C=1)
            image_type: List of corresponding class image_type
            save_png: Whether to save PNG image files to the specified directory

        Note:
            • Use NPZ compressed format to save storage space
            • Optionally save PNG files for easier visualization
            • Include complete data validation and statistics
            • Support efficient storage for large-scale datasets
        """
        print(f"Saving augmented data to: {self.all_npz}")

        # Data format validation
        if len(images) != len(masks) or len(images) != len(image_type):
            raise ValueError(f"Data length mismatch: images={len(images)}, masks={len(masks)}, image_type={len(image_type)}")

        # Convert to numpy arrays
        images_array = np.array(images, dtype=np.float32)  # (N, H, W) single-channel grayscale
        masks_array = np.array(masks, dtype=np.uint8)      # (N, H, W)
        

        image_type_array = np.array(image_type, dtype='<U10')  

        filenames_array = np.array(filenames, dtype='<U100')  # Store filenames for reference
        
        # Data shape validation
        expected_img_shape = (len(images), 1, self.target_size[0], self.target_size[1])
        expected_mask_shape = (len(masks), 1, self.target_size[0], self.target_size[1])

        if images_array.shape != expected_img_shape:
            raise ValueError(f"Image array shape mismatch: {images_array.shape} vs {expected_img_shape}")
        if masks_array.shape != expected_mask_shape:
            raise ValueError(f"Mask array shape mismatch: {masks_array.shape} vs {expected_mask_shape}")

        # Data range validation
        img_min, img_max = images_array.min(), images_array.max()
        mask_min, mask_max = masks_array.min(), masks_array.max()

        assert img_max <= 1, f"Image value out of range: {img_max}"
        assert img_min >= -1, f"Image value below range: {img_min}"
        assert mask_max <= 1, f"Mask value out of range: {mask_max}"
        assert mask_min >= 0, f"Mask value below range: {mask_min}"
        
        # Save NPZ file
        self.__save_npz_files(images_array, masks_array, image_type_array)
        # Save PNG files (if enabled)
        if save_png:
            self.__save_png_files(images_array, masks_array, image_type_array, filenames_array)

        # Count samples per class
        unique_image_type, counts = np.unique(image_type_array, return_counts=True)
        print('=' * 50)
        print(f"Final distribution: \n", end="")
        for label, count in zip(unique_image_type, counts):
            percentage = count / len(image_type_array) * 100
            print(f"{label}={count}({percentage:.1f}%)\n", end="")
        print('=' * 50)

    def __save_npz_files(self, images_array: np.ndarray,
                         masks_array: np.ndarray,
                         image_type_array: np.ndarray) -> None:
        np.savez(  # use np.savez to save multiple arrays in a single file
            self.all_npz,
            images=images_array,
            masks=masks_array,
            image_type=image_type_array,
            # addtional metadata
            metadata=np.array([{
                'target_size': self.target_size,
                'num_samples': len(images_array),
                'creation_time': pd.Timestamp.now().isoformat(),
                'aug_config': self.target_aug_times
            }], dtype=object)
        )
        
        file_size = self.all_npz.stat().st_size / (1024**2)
        print(f"NPZ file saved: {file_size:.1f} MB, {len(image_type_array)} samples")

    def __save_png_files(self, images_array: np.ndarray,
                         masks_array: np.ndarray,
                         image_type_array: np.ndarray,
                         filenames_array: np.ndarray) -> None:
        """
        Save augmented images and masks as PNG files

        Args:
            images_array: Image array (N, 1, H, W) - single-channel grayscale images
            masks_array: Mask array (N, 1, H, W) - single-channel binary masks
            image_type_array: Label array (N,)
        """
        total_samples = len(images_array)
        print(f"Saving {total_samples} PNG files...")

        # Count samples per class for naming
        class_counters = {}
        for label in np.unique(image_type_array):
            class_counters[label] = 0

        # Save PNG files one by one
        for i in tqdm(range(total_samples), desc="Saving PNG files", disable=False):
            try:
                # Get current sample data
                image = images_array[i]  # (1, H, W) - single-channel grayscale image
                mask = masks_array[i]    # (1, H, W) - single-channel binary mask
                label = image_type_array[i]  # str

                # Update class counter
                class_counters[label] += 1

                # Generate file name
                image_filename = f"{filenames_array[i]}.png"
                mask_filename = f"{filenames_array[i]}_mask.png"

                image_path = self.images_aug_dir / image_filename
                mask_path = self.masks_aug_dir / mask_filename
                
                # Handle single-channel image: (1, H, W) -> (H, W)
                image_hw = image[0]  # (H, W) extract single channel
                
                # Adjust image data range
                if image_hw.min() >= -1.1 and image_hw.max() <= 1.1:
                    # If image data is in [-1, 1] range convert to [0, 255]
                    if image_hw.min() < -0.1:
                        image_hw = (image_hw + 1.0) / 2.0
                    # Convert to [0, 255]
                    image_hw = (image_hw * 255).clip(0, 255).astype(np.uint8)
                else:
                    # If image data is already in [0, 255] range
                    image_hw = image_hw.clip(0, 255).astype(np.uint8)
                
                # Handle single-channel mask: (1, H, W) -> (H, W)
                mask_hw = mask[0]  # (H, W)
                # Ensure mask values are in [0, 255] range
                mask_hw = (mask_hw * 255).astype(np.uint8)

                # Save image and mask
                cv2.imwrite(str(image_path), image_hw)
                cv2.imwrite(str(mask_path), mask_hw)
                
            except Exception as e:
                print(f"\nWarning: Error saving PNG file for sample {i+1}: {e}")
                continue
        
        # Final report
        print(f"PNG files saved to: {self.images_aug_dir} and {self.masks_aug_dir}")

class DataSplit(DataInfo):
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        super().__init__()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_data_with_save(self, enable_balance: bool = True) -> None:
        """
        Split the dataset into train, validation, and test sets, and save the splits to NPZ files.
        """
        # self.target_aug_times decide how many time to augment each image
        # self.seed is used for reproducibility
        # if enable balance is True, try to get the negative samples percentage in the whole dataset
        
        # Load augmented data from NPZ file
        if not self.all_npz.exists():
            raise FileNotFoundError(f"Augmented NPZ file not found: {self.all_npz}")
            
        print(f"Loading augmented data from: {self.all_npz}")
        images, masks, image_type = self.read_npz_file(self.all_npz)
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        # Get unique image_type and their counts
        unique_image_type, label_counts = np.unique(image_type, return_counts=True)
        print(f"Dataset distribution before split:")
        for label, count in zip(unique_image_type, label_counts):
            percentage = count / len(image_type) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Create indices for each class
        class_indices = {}
        for label in unique_image_type:
            class_indices[label] = np.where(image_type == label)[0]
            # Shuffle indices for each class
            np.random.shuffle(class_indices[label])
        
        # Split indices for each class
        train_indices = []
        val_indices = []
        test_indices = []
        
        if enable_balance:
            # Balanced split: maintain class distribution across splits
            print("Using balanced split strategy")
            for label in unique_image_type:
                indices = class_indices[label]
                n_samples = len(indices)
                
                # Calculate split sizes
                n_train = int(n_samples * self.train_ratio)
                n_val = int(n_samples * self.val_ratio)
                n_test = n_samples - n_train - n_val  # Remaining samples go to test
                
                # Split indices
                train_indices.extend(indices[:n_train])
                val_indices.extend(indices[n_train:n_train + n_val])
                test_indices.extend(indices[n_train + n_val:])
                
                print(f"  {label}: train={n_train}, val={n_val}, test={n_test}")
        else:
            # Random split: shuffle all indices and split
            print("Using random split strategy")
            all_indices = np.arange(len(image_type))
            np.random.shuffle(all_indices)
            
            # Calculate split sizes
            n_total = len(all_indices)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)
            n_test = n_total - n_train - n_val
            
            # Split indices
            train_indices = all_indices[:n_train]
            val_indices = all_indices[n_train:n_train + n_val]
            test_indices = all_indices[n_train + n_val:]
            
            print(f"Random split: train={n_train}, val={n_val}, test={n_test}")
        
        # Convert to numpy arrays and shuffle within each split
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)
        
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        
        # Extract data for each split
        train_images = images[train_indices]
        train_masks = masks[train_indices]
        train_image_type = image_type[train_indices]
        
        val_images = images[val_indices]
        val_masks = masks[val_indices]
        val_image_type = image_type[val_indices]
        
        test_images = images[test_indices]
        test_masks = masks[test_indices]
        test_image_type = image_type[test_indices]
        
        # Print final distribution for each split
        print("\nFinal split distribution:")
        for split_name, split_image_type in [("Train", train_image_type), ("Val", val_image_type), ("Test", test_image_type)]:
            unique_split_image_type, split_counts = np.unique(split_image_type, return_counts=True)
            print(f"{split_name} ({len(split_image_type)} samples):")
            for label, count in zip(unique_split_image_type, split_counts):
                percentage = count / len(split_image_type) * 100
                print(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Save splits to NPZ files
        self._save_split_to_npz(train_images, train_masks, train_image_type, self.train_npz, "train")
        self._save_split_to_npz(val_images, val_masks, val_image_type, self.val_npz, "validation")
        self._save_split_to_npz(test_images, test_masks, test_image_type, self.test_npz, "test")
        
        print(f"\nData split completed successfully!")
        print(f"Train set saved to: {self.train_npz}")
        print(f"Validation set saved to: {self.val_npz}")
        print(f"Test set saved to: {self.test_npz}")
    
    def _save_split_to_npz(self, images: np.ndarray, masks: np.ndarray, image_type: np.ndarray, 
                          npz_path: Path, split_name: str) -> None:
        """
        Save a data split to NPZ file with metadata.
        """
        target_size = (self.max_height, self.max_width)  # (height, width)
        # Create metadata for this split
        metadata = np.array([{
            'split_type': split_name,
            'target_size': target_size,
            'num_samples': len(images),
            'creation_time': pd.Timestamp.now().isoformat(),
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'seed': self.seed
        }], dtype=object)
        
        # Save to NPZ file
        np.savez(
            npz_path,
            images=images,
            masks=masks,
            image_type=image_type,
            metadata=metadata
        )
        
        # Calculate and print file size
        file_size = npz_path.stat().st_size / (1024**2)
        print(f"{split_name.capitalize()} NPZ saved: {file_size:.1f} MB, {len(image_type)} samples")

    @staticmethod
    def read_npz_file(npz_file: Path) -> tuple:
        """
        Read NPZ file and return images, masks, image_type, and metadata.
        """
        if not npz_file.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_file}")

        data = np.load(npz_file, allow_pickle=True)
        images = data['images']
        masks = data['masks']
        image_type = data['image_type']
        
        # Print metadata if available
        # if 'metadata' in data.files:
        #     metadata = data['metadata']
        #     # Only [0] is used, as metadata is a single-element array
        #     md = metadata[0]   
        #     # Traversal and print metadata
        #     for key, value in md.items():
        #         print(f"{key}: {value}")

        return images, masks, image_type

class ReadDataset(Dataset):
    """
    Custom dataset class to read data from NPZ files.
    """
    def __init__(self, dataset_type: str = 'train', batch_size: int = 8):
        super().__init__()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.datainfo = DataInfo()

        # Load the appropriate NPZ file based on dataset type
        if dataset_type == 'train':
            self.data_path = self.datainfo.train_npz
        elif dataset_type == 'validation':
            self.data_path = self.datainfo.val_npz
        elif dataset_type == 'test':
            self.data_path = self.datainfo.test_npz
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        # Read data from NPZ file
        self.images, self.masks, self.image_type = DataSplit().read_npz_file(self.data_path)
        
        print(f"Loaded {self.dataset_type} dataset: {len(self.images)} samples")
        
        # Validate data shapes
        unique_types, counts = np.unique(self.image_type, return_counts=True)
        for img_type, count in zip(unique_types, counts):
            percentage = count / len(self.image_type) * 100
            print(f"  {img_type}: {count} ({percentage:.1f}%)")
    
    def __len__(self):
        """return the number of samples in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")
        
        # Get the image, mask, and label for the given index
        image = self.images[idx]  # (1, H, W)
        mask = self.masks[idx]    # (1, H, W)
        label = self.image_type[idx]
        
        # convert to PyTorch tensors
        image_tensor = torch.from_numpy(image).float()  # (1, H, W)
        mask_tensor = torch.from_numpy(mask).float()    # (1, H, W)

        # Ensure the tensor input is in 3 dimensions (C, H, W) and C is 1
        assert image_tensor.ndim == 3 and image_tensor.shape[0] == 1 , f"Image tensor shape error: {image_tensor.shape}"
        assert mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1, f"Mask tensor shape error: {mask_tensor.shape}"
        
        return image_tensor, mask_tensor, label
    
    def get_dataloader(self, shuffle: bool = None, num_workers: int = None, drop_last: bool = False):
        """
        return a DataLoader for the dataset
        
        Args:
            shuffle: whether to shuffle the dataset, default is True for training, False for validation/test
            num_workers: data loading workers, default is 0 for Windows, 4 for Linux
            drop_last: whether to drop the last incomplete batch, default is False
            
        Returns:
            DataLoader object
        """
        # Default settings: shuffle training set, don't shuffle validation/test sets
        if shuffle is None:
            shuffle = (self.dataset_type == 'train')
        
        # adjust num_workers based on platform
        if num_workers is None:
            num_workers = 0 if sys.platform.startswith('win') else 4
        
        dataloader = DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available() 
        )
        
        return dataloader
    
    def get_class_weights(self):
        """
        calculate class weights based on the distribution of image types
        Returns:
            dict: {class_name: weight}
        """
        unique_types, counts = np.unique(self.image_type, return_counts=True)
        total_samples = len(self.image_type)
        
        # 计算类别权重 (inverse frequency)
        class_weights = {}
        for img_type, count in zip(unique_types, counts):
            weight = total_samples / (len(unique_types) * count)
            class_weights[img_type] = weight
        return class_weights
      
                 

if __name__ == "__main__":

    # False and True to run data augmentation and split
    run_data_augmentation = True
    run_data_split = True
    test_dataloader = False  

    if run_data_augmentation:
        aug_dataset = AugmentedDataset()
        images, masks, image_type, filenames = aug_dataset.generate_augmented_data()
        aug_dataset.save_augmented_data(images, masks, image_type, filenames, save_png=True)
    
    if run_data_split:
        data_splitter = DataSplit()
        # Enable balanced split to maintain class distribution across splits
        data_splitter.split_data_with_save(enable_balance=True)
    
    if test_dataloader:
        print("=" * 50)
        print("Testing DataLoader functionality...")
        print("=" * 50)
        
        try:
            # Create DataLoader instances for train, validation, and test sets
            train_dataset = ReadDataset(dataset_type='train', batch_size=32)

            train_loader = train_dataset.get_dataloader()
            
            # test train DataLoader
            print("\nTesting train loader...")
            for i, (images, masks, image_type) in enumerate(train_loader):
                print(f"Batch {i+1}: Images shape: {images.shape}, Masks shape: {masks.shape}")
                print(f"Labels: {image_type}")
                if i >= 2:
                    break
            
            print("\nDataLoader test completed successfully!")
            
        except Exception as e:
            print(f"DataLoader test failed: {e}")
            import traceback
            traceback.print_exc()
