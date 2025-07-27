import os
import sys
import math
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from DataInfo import DataInfo
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
            # resize back to target size
            A.Resize(self.target_size[0], self.target_size[1], 
            interpolation=cv2.INTER_AREA, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
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
    
    def load_image_and_mask(self, image_path, mask_path):
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
    
    def crop_image(self, img: np.ndarray,
                   crop_top: int = 100,
                   crop_bottom: int = 100,
                   crop_left: int = 100,
                   crop_right: int = 100) -> np.ndarray:
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
        
        for i in range(num_augmentations):
            try:
                # Use the SAME cropped image for each augmentation iteration
                # apply random data augmentation transformations
                augmented = self.transform(image=cropped_image, mask=cropped_mask)
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
            print(f"  {class_name}: {count} images -> {count * target_aug} images after augmentation")

        # Store augmentation results
        all_images = []
        all_masks = []
        all_labels = []
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

                # perform data augmentation
                aug_images, aug_masks = self.augment_single_image(image, mask, aug_times)

                # Add to the total dataset
                all_images.extend(aug_images)
                all_masks.extend(aug_masks)
                all_labels.extend([mask_status] * len(aug_images))
                image_name = image_name.split('.')[0]  # Remove file extension for consistency
                all_filenames.extend(f"{image_name}_{i:04d}" for i in range(len(aug_images)))

                # Update statistics
                success_count += 1
                class_generated[mask_status] += len(aug_images)
                
            except Exception as e:
                error_count += 1
                raise RuntimeError(f"Error processing sample {idx+1}: {e}")
        
        print("\n" + "=" * 60)
        print("Data augmentation statistics:")
        print("=" * 60)
        print(f"Successfully processed: {success_count}/{len(df)} original samples")
        print(f"Skipped samples: {skip_count}")
        print(f"Failed samples: {error_count}")
        print(f"Total generated images: {len(all_images)}")

        print(f"Class-wise generated statistics:")
        for class_name, generated_count in class_generated.items():
            original_count = class_counts.get(class_name, 0)
            expected_count = original_count * self.target_aug_times[class_name]
            print(f"  {class_name}: {generated_count}/{expected_count} images "
                    f"({generated_count/expected_count*100:.1f}%)")
        
        if len(all_images) == 0:
            raise RuntimeError("No augmented data was successfully generated!")

        return all_images, all_masks, all_labels, all_filenames

    def save_augmented_data(self, images, masks, labels, filenames, save_png=True):
        """
        Save augmented data to NPZ compressed file, with optional PNG image file saving
        
        Args:
            images: List of augmented images (each with shape CxHxW)
            masks: List of augmented masks (each with shape HxW, converted to CxHxW where C=1)
            labels: List of corresponding class labels
            save_png: Whether to save PNG image files to the specified directory

        Note:
            • Use NPZ compressed format to save storage space
            • Optionally save PNG files for easier visualization
            • Include complete data validation and statistics
            • Support efficient storage for large-scale datasets
        """
        print(f"\nSaving augmented data to: {self.all_npz}")
        if save_png:
            print(f"Also saving PNG files to: {self.images_aug_dir} and {self.masks_aug_dir}")

        # Data format validation
        if len(images) != len(masks) or len(images) != len(labels):
            raise ValueError(f"Data length mismatch: images={len(images)}, masks={len(masks)}, labels={len(labels)}")

        # Convert to numpy arrays
        print("Converting data format...")
        images_array = np.array(images, dtype=np.float32)  # (N, H, W) single-channel grayscale
        masks_array = np.array(masks, dtype=np.uint8)      # (N, H, W)
        

        labels_array = np.array(labels, dtype='<U10')  

        filenames_array = np.array(filenames, dtype='<U100')  # Store filenames for reference
        
        # Data shape validation
        expected_img_shape = (len(images), 1, self.target_size[0], self.target_size[1])
        expected_mask_shape = (len(masks), 1, self.target_size[0], self.target_size[1])

        if images_array.shape != expected_img_shape:
            raise ValueError(f"Image array shape mismatch: {images_array.shape} vs {expected_img_shape}")
        if masks_array.shape != expected_mask_shape:
            raise ValueError(f"Mask array shape mismatch: {masks_array.shape} vs {expected_mask_shape}")

        # Data range validation
        print("Validating data quality...")
        img_min, img_max = images_array.min(), images_array.max()
        mask_min, mask_max = masks_array.min(), masks_array.max()

        print(f"Image data range: [{img_min:.3f}, {img_max:.3f}]")
        print(f"Mask data range: [{mask_min}, {mask_max}]")

        assert mask_max <= 1, f"Mask value out of range: {mask_max}"
        assert mask_min >= 0, f"Mask value below range: {mask_min}"

        # Estimate storage size
        total_size_mb = (images_array.nbytes + masks_array.nbytes + labels_array.nbytes) / (1024**2)
        print(f"Estimated storage size: {total_size_mb:.1f} MB")

        # Save PNG files (if enabled)
        if save_png:
            print("\nSaving PNG files...")
            self._save_png_files(images_array, masks_array, labels_array, filenames_array)

        # Save as NPZ file
        print("Saving NPZ data...")
        print(f"Data size: Images {images_array.nbytes/1024**2:.1f}MB, Masks {masks_array.nbytes/1024**2:.1f}MB")

        # Save data - use uncompressed format for speed
        print("Writing files (using fast save mode)...")
        start_time = pd.Timestamp.now()
        
        np.savez(  # 使用 savez 而不是 savez_compressed 提高速度
            self.all_npz,
            images=images_array,
            masks=masks_array,
            labels=labels_array,
            # 额外的元数据
            metadata=np.array([{
                'target_size': self.target_size,
                'num_samples': len(images),
                'creation_time': pd.Timestamp.now().isoformat(),
                'aug_config': self.target_aug_times
            }], dtype=object)
        )
        
        save_time = pd.Timestamp.now() - start_time
        print(f"Saving NPZ file took: {save_time.total_seconds():.1f} seconds")

        # 验证保存结果
        actual_size_mb = self.all_npz.stat().st_size / (1024**2)
        size_ratio = actual_size_mb / total_size_mb * 100
        
        print("\n" + "=" * 50)
        print("Data saving completed!")
        print("=" * 50)
        print(f"File path: {self.all_npz}")
        print(f"Actual size: {actual_size_mb:.1f} MB (take {size_ratio:.1f}% of estimated size)")
        print(f"Image shape: {images_array.shape}")
        print(f"Mask shape: {masks_array.shape}")
        print(f"Total samples: {len(labels_array)}")

        # Count samples per class
        print(f"\nClass distribution:")
        unique_labels, counts = np.unique(labels_array, return_counts=True)
        for label, count in zip(unique_labels, counts):
            percentage = count / len(labels_array) * 100
            print(f"  {label}: {count} images ({percentage:.1f}%)")

    def _save_png_files(self, images_array, masks_array, labels_array, filenames_array):
        """
        Save augmented images and masks as PNG files

        Args:
            images_array: Image array (N, 1, H, W) - single-channel grayscale images
            masks_array: Mask array (N, 1, H, W) - single-channel binary masks
            labels_array: Label array (N,)
        """
        total_samples = len(images_array)
        print(f"Saving {total_samples} samples as PNG files...")

        # Count samples per class for naming
        class_counters = {}
        for label in np.unique(labels_array):
            class_counters[label] = 0

        # Save PNG files one by one
        for i in tqdm(range(total_samples), desc="Saving PNG files", disable=False):
            try:
                # Get current sample data
                image = images_array[i]  # (1, H, W) - single-channel grayscale image
                mask = masks_array[i]    # (1, H, W) - single-channel binary mask
                label = labels_array[i]  # str

                # Update class counter
                class_counters[label] += 1
                sample_id = class_counters[label]

                # Generate file name
                image_filename = f"{filenames_array[i]}.png"
                mask_filename = f"{filenames_array[i]}_mask.png"

                image_path = self.images_aug_dir / image_filename
                mask_path = self.masks_aug_dir / mask_filename
                
                # Deal with single-channel image: (1, H, W) -> (H, W)
                image_hw = image[0]  # (H, W) 取出单通道
                
                # revise image data range
                if image_hw.min() >= -1.1 and image_hw.max() <= 1.1:
                    # if image data is in [-1, 1] range convert to [0, 255]
                    if image_hw.min() < -0.1:
                        image_hw = (image_hw + 1.0) / 2.0
                    # convert to [0, 255]
                    image_hw = (image_hw * 255).clip(0, 255).astype(np.uint8)
                else:
                    # if image data is already in [0, 255] range
                    image_hw = image_hw.clip(0, 255).astype(np.uint8)
                
                # Deal with single-channel mask: (1, H, W) -> (H, W)
                mask_hw = mask[0]  # (H, W)
                # Ensure mask values are in [0, 255] range
                mask_hw = (mask_hw * 255).astype(np.uint8)

                # Save image and mask
                cv2.imwrite(str(image_path), image_hw)
                cv2.imwrite(str(mask_path), mask_hw)
                
            except Exception as e:
                print(f"\nWarning: there was an error saving PNG file for sample {i+1}: {e}")
                continue
        
        # Final report
        print(f"\nFinished saving {total_samples} PNG files.")
        print(f"Images saved to: {self.images_aug_dir}")
        print(f"Masks saved to: {self.masks_aug_dir}")

class ReadDataset(Dataset):
    
    def __init__(self, dataset_type='all', batch_size=16, auto_load=True, balanced_sampling=False):
        # Get configuration information from the Dataset base class
        self.config = DataInfo()
        
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.balanced_sampling = balanced_sampling
        self.valid_types = ['all', 'train', 'val', 'test']
        
        if dataset_type not in self.valid_types:
            raise ValueError(f"Dataset type must be one of {self.valid_types}, current: {dataset_type}")

        self.npz_files = {
            'all': self.config.all_npz,
            'train': self.config.train_npz,
            'val': self.config.val_npz,
            'test': self.config.test_npz
        }
        
        self.current_file = self.npz_files[dataset_type]
        self.data = None
        self.images = None
        self.masks = None
        self.labels = None
        self.metadata = None
        self.length = 0
        
        self.class_weights = None
        self.sample_weights = None
        self.label_to_idx = None

        print(f"Initializing ReadDataset - Type: {dataset_type}, Batch size: {batch_size}")
        print(f"Target file: {self.current_file}")
        print(f"Balanced sampling: {'Enabled' if balanced_sampling else 'Disabled'}")

        # Auto load data
        if auto_load:
            self.load_data()
    
    def load_data(self):
        """
        Load data from the specified NPZ file
        
        Returns:
            tuple: (images, masks, labels) - Loaded data

        Raises:
            FileNotFoundError: When the NPZ file does not exist
            KeyError: When the NPZ file is missing required keys
        """
        if not self.current_file.exists():
            raise FileNotFoundError(f"Data file does not exist: {self.current_file}")

        print(f"\nLoading data file: {self.current_file}")
        start_time = pd.Timestamp.now()
        
        try:
            # Load NPZ file
            self.data = np.load(self.current_file, allow_pickle=True)

            # Extract data
            self.images = self.data['images']
            self.masks = self.data['masks']
            self.labels = self.data['labels']
            self.length = len(self.images)
            
            # try to extract metadata if available
            try:
                self.metadata = self.data['metadata'].item() if 'metadata' in self.data else None
            except:
                self.metadata = None
                
            load_time = pd.Timestamp.now() - start_time
            print(f"Data loading complete! Time taken: {load_time.total_seconds():.2f} seconds")

            # Display basic information
            self.print_basic_info()

            # If balanced sampling is enabled, calculate sample weights
            if self.balanced_sampling:
                self._calculate_sample_weights()
            
            return self.images, self.masks, self.labels
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data file: {e}")

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        PyTorch Dataset getitem method
        
        Args:
            idx: Sample index

        Returns:
            tuple: (image_tensor, mask_tensor, label_str) - Data for a single sample
                - image_tensor: torch.Tensor, shape (1, H, W), dtype=float32 - Single-channel grayscale image
                - mask_tensor: torch.Tensor, shape (1, H, W), dtype=float32 - Single-channel binary mask
                - label_str: str, Label string
        """
        if self.data is None:
            raise RuntimeError("Data not loaded, please call load_data() or set auto_load=True")

        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index out of range: {idx}, valid range: [0, {self.length-1}]")

        # Get numpy data
        image = self.images[idx]  # (1, H, W) - Single-channel image
        mask = self.masks[idx]    # (1, H, W) - Single-channel mask
        label = self.labels[idx]  # str

        # Validate data format
        assert image.shape[0] == 1, f"Image should be single-channel, current shape: {image.shape}"
        assert mask.shape[0] == 1, f"Mask should be single-channel, current shape: {mask.shape}"

        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image.copy()).float()  # (1, H, W)
        mask_tensor = torch.from_numpy(mask.copy()).float()    # (1, H, W)
        
        return image_tensor, mask_tensor, label
    
    def _calculate_sample_weights(self):
        """
        Calculate sample weights for balanced sampling

        Principles:
        1. Count the number of samples for each class
        2. Calculate class weight = 1 / (number of samples in class / total number of samples)
        3. Assign the corresponding class weight to each sample
        4. Samples from minority classes will receive higher sampling weights
        """
        print(f"\nCalculating sample weights for balanced sampling...")

        # Count the number of samples for each class
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)

        # Calculate class weights (minority classes receive higher weights)
        self.class_weights = {}
        for label, count in zip(unique_labels, counts):
            # Weight = total_samples / (num_classes * count)
            weight = total_samples / (len(unique_labels) * count)
            self.class_weights[label] = weight

        # Assign weights to each sample
        self.sample_weights = np.array([self.class_weights[label] for label in self.labels])

        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        print(f"Category weights calculation completed:")
        for label, weight in self.class_weights.items():
            count = counts[list(unique_labels).index(label)]
            percentage = count / total_samples * 100
            print(f"  {label}: Sample count={count} ({percentage:.1f}%), Weight={weight:.3f}")

        print(f"Balanced sampling configuration:")
        print(f"  Sample weight range: [{self.sample_weights.min():.3f}, {self.sample_weights.max():.3f}]")
        print(f"  Weight ratio: {self.sample_weights.max()/self.sample_weights.min():.2f}:1")

    def get_dataloader(self, shuffle=True, num_workers=None, pin_memory=None):
        """
        Get a PyTorch DataLoader for the dataset
        
        Args:
            shuffle: Whether to shuffle the data (ignored when using balanced sampling)
            num_workers: Number of worker processes (auto-detect if None: 0 for Windows, 4 for others)
            pin_memory: Whether to use pinned memory (auto-detect if None: True for CUDA, False for CPU/MPS)
            
        Returns:
            DataLoader: PyTorch DataLoader instance
        """
        # Auto-detect num_workers based on platform to avoid Windows multiprocessing issues
        if num_workers is None:
            num_workers = 0 if sys.platform == 'win32' else 4
            
        # Auto-detect pin_memory based on device to avoid warnings
        if pin_memory is None:
            # Only use pin_memory for CUDA devices to avoid warnings
            pin_memory = torch.cuda.is_available()
            
        if self.balanced_sampling and self.sample_weights is not None:
            # 使用加权随机采样器实现平衡采样
            print(f"\nUsing balanced sampling mode with WeightedRandomSampler")
            print(f"DataLoader settings: num_workers={num_workers}, pin_memory={pin_memory}")
            sampler = WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True  # Allow resampling
            )
            
            return DataLoader(
                self,
                batch_size=self.batch_size,
                sampler=sampler,  # Use sampler instead of shuffle
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        else:
            # Use standard random sampling
            print(f"\nUsing standard random sampling mode")
            print(f"DataLoader settings: num_workers={num_workers}, pin_memory={pin_memory}")
            return DataLoader(
                self, 
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
    
    def print_basic_info(self):
        """
        Print basic information about the dataset
        """
        if self.data is None:
            print("请先调用 load_data() 加载数据")
            return
        
        print("\n" + "=" * 50)
        print(f"数据集信息 - {self.dataset_type.upper()}")
        print("=" * 50)
        
        # 文件信息
        file_size_mb = self.current_file.stat().st_size / (1024**2)
        print(f"文件路径: {self.current_file}")
        print(f"文件大小: {file_size_mb:.1f} MB")
        
        # 数据基本信息
        print(f"\n数据概览:")
        print(f"  样本总数: {len(self.images)}")
        print(f"  图像形状: {self.images.shape}")
        print(f"  掩码形状: {self.masks.shape}")
        print(f"  标签数量: {len(self.labels)}")
        print(f"  批次大小: {self.batch_size}")
        
        # 数据类型和范围
        print(f"\n数据类型:")
        print(f"  图像类型: {self.images.dtype}")
        print(f"  掩码类型: {self.masks.dtype}")
        print(f"  标签类型: {self.labels.dtype}")
        
        print(f"\n数据范围:")
        print(f"  图像范围: [{self.images.min():.3f}, {self.images.max():.3f}]")
        print(f"  掩码范围: [{self.masks.min()}, {self.masks.max()}]")
        
        # 标签分布
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"\n标签分布:")
        for label, count in zip(unique_labels, counts):
            percentage = count / len(self.labels) * 100
            print(f"  {label}: {count} 张 ({percentage:.1f}%)")
        
        # DataLoader信息
        total_batches = (len(self.images) + self.batch_size - 1) // self.batch_size
        print(f"\nDataLoader信息:")
        print(f"  总批次数: {total_batches}")
        print(f"  最后批次大小: {len(self.images) % self.batch_size or self.batch_size}")
        print(f"  平衡采样: {'开启' if self.balanced_sampling else '关闭'}")
        
        if self.balanced_sampling and self.sample_weights is not None:
            print(f"  采样模式: 加权随机采样 (WeightedRandomSampler)")
            print(f"  每个batch中各类别样本期望比例: 接近 1:1")
        else:
            print(f"  采样模式: 标准随机采样")
            print(f"  每个batch中各类别样本比例: 随机分布")
        
        # 元数据信息
        if self.metadata:
            print(f"\n元数据信息:")
            for key, value in self.metadata.items():
                print(f"  {key}: {value}")
        
        print("=" * 50)
    
    
    def close(self):
        """
        Dispose of the dataset and release memory
        """
        if self.data is not None:
            self.data.close()
            self.data = None
            self.images = None
            self.masks = None
            self.labels = None
            self.metadata = None
            self.length = 0
            self.class_weights = None
            self.sample_weights = None
            self.label_to_idx = None
            print("Dataset resources have been released.")

class DataSplit(DataInfo):

    def __init__(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Data splitter for ultrasound image datasets
        
        Args:
            train_ratio: train set ratio (default 0.7)
            val_ratio: validation set ratio (default 0.15)
            test_ratio: test set ratio (default 0.15)
            seed: random seed for reproducibility, inherited from DataInfo class

        Raises:
            ValueError: not summing to 1.0
        """
        super().__init__()
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Training, validation and test set ratios must sum to 1, current: {train_ratio + val_ratio + test_ratio}")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 设置随机种子确保可重现
        np.random.seed(self.seed)
        random.seed(self.seed)

        print(f"Initializing dataset splitter:")
        print(f"  Train set ratio: {train_ratio:.1%}")
        print(f"  Validation set ratio: {val_ratio:.1%}")
        print(f"  Test set ratio: {test_ratio:.1%}")
        print(f"  Random seed: {self.seed}")

    def split_and_save_dataset(self, source_npz_path=None):
        """
        Split the dataset and save it as three NPZ files

        Args:
            source_npz_path: source NPZ file path, if None, use self.all_npz

        Process:
            1. Load the source dataset
            2. Perform stratified sampling by label
            3. Ensure consistent class proportions in train/val/test sets
            4. Save as three separate NPZ files
        """
        # 确定源文件路径
        if source_npz_path is None:
            source_npz_path = self.all_npz
        else:
            source_npz_path = Path(source_npz_path)
            
        if not source_npz_path.exists():
            raise FileNotFoundError(f"Source data file does not exist: {source_npz_path}")

        print("=" * 60)
        print("Starting dataset splitting process")
        print("=" * 60)
        print(f"Source file: {source_npz_path}")

        # Load the source data
        print(f"\nLoading source data...")
        start_time = pd.Timestamp.now()
        data = np.load(source_npz_path, allow_pickle=True)
        
        images = data['images']
        masks = data['masks']
        labels = data['labels']
        metadata = data.get('metadata', None)
        
        total_samples = len(images)
        load_time = pd.Timestamp.now() - start_time
        print(f"Data loading complete! Time taken: {load_time.total_seconds():.2f} seconds")
        print(f"Total samples: {total_samples}")

        # Statistics of the original data distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\nOriginal data distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = count / total_samples * 100
            print(f"  {label}: {count} images ({percentage:.1f}%)")

        # Stratified sampling to split the dataset
        train_indices, val_indices, test_indices = self._stratified_split(labels)

        # Create training set
        train_images = images[train_indices]
        train_masks = masks[train_indices]
        train_labels = labels[train_indices]

        # Create validation set
        val_images = images[val_indices]
        val_masks = masks[val_indices]
        val_labels = labels[val_indices]

        # Create test set
        test_images = images[test_indices]
        test_masks = masks[test_indices]
        test_labels = labels[test_indices]

        # Validate split results
        self._validate_split(train_labels, val_labels, test_labels)

        # Save datasets
        datasets = {
            'train': (train_images, train_masks, train_labels, self.train_npz),
            'val': (val_images, val_masks, val_labels, self.val_npz),
            'test': (test_images, test_masks, test_labels, self.test_npz)
        }
        
        for split_name, (split_images, split_masks, split_labels, split_path) in datasets.items():
            self._save_split_dataset(
                split_images, split_masks, split_labels, 
                split_path, split_name, metadata
            )
        
        # 关闭源数据文件
        data.close()
        
        print("\n" + "=" * 60)
        print("Dataset splitting complete!")
        print("=" * 60)
        print("Generated files:")
        for split_name, (_, _, _, split_path) in datasets.items():
            file_size = split_path.stat().st_size / (1024**2)
            print(f"  {split_name:>5}: {split_path} ({file_size:.1f} MB)")
        
        return {
            'train': (train_images, train_masks, train_labels),
            'val': (val_images, val_masks, val_labels),
            'test': (test_images, test_masks, test_labels)
        }
    
    def _stratified_split(self, labels):
        """
        Perform stratified sampling to split dataset indices

        Args:
            labels: Label array

        Returns:
            tuple: (train_indices, val_indices, test_indices)
            
        Note:
            Ensure that the proportion of each class in the train/val/test sets is consistent with the overall proportion
        """
        print(f"\nPerforming stratified sampling...")
        
        # Get indices of all samples
        all_indices = np.arange(len(labels))

        # Group by label
        unique_labels = np.unique(labels)
        train_indices = []
        val_indices = []
        test_indices = []
        
        for label in unique_labels:
            # Get all indices for the current label
            label_indices = all_indices[labels == label]
            label_count = len(label_indices)

            # Shuffle indices randomly
            np.random.shuffle(label_indices)

            # Calculate the number of samples for each dataset
            train_count = int(label_count * self.train_ratio)
            val_count = int(label_count * self.val_ratio)
            test_count = label_count - train_count - val_count  # Ensure all samples are assigned

            # Assign indices
            train_end = train_count
            val_end = train_end + val_count
            
            train_indices.extend(label_indices[:train_end])
            val_indices.extend(label_indices[train_end:val_end])
            test_indices.extend(label_indices[val_end:])
            
            print(f"  {label}: ALL{label_count} -> train{train_count}, val{val_count}, test{test_count}")
        
        # Convert to numpy arrays and shuffle
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)
        
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        
        print(f"分层采样完成:")
        print(f"  训练集: {len(train_indices)} 个样本")
        print(f"  验证集: {len(val_indices)} 个样本")
        print(f"  测试集: {len(test_indices)} 个样本")
        
        return train_indices, val_indices, test_indices
    
    def _validate_split(self, train_labels, val_labels, test_labels):
        """
        验证数据集划分的正确性
        
        Args:
            train_labels: 训练集标签
            val_labels: 验证集标签
            test_labels: 测试集标签
        """
        print(f"\n验证数据集划分:")
        
        datasets = {
            '训练集': train_labels,
            '验证集': val_labels,
            '测试集': test_labels
        }
        
        # 统计各数据集的标签分布
        all_stats = {}
        for dataset_name, dataset_labels in datasets.items():
            unique_labels, counts = np.unique(dataset_labels, return_counts=True)
            total_count = len(dataset_labels)
            
            stats = {}
            for label, count in zip(unique_labels, counts):
                percentage = count / total_count * 100
                stats[label] = {'count': count, 'percentage': percentage}
            
            all_stats[dataset_name] = stats
            
            print(f"  {dataset_name} ({total_count} 个样本):")
            for label, stat in stats.items():
                print(f"    {label}: {stat['count']} 张 ({stat['percentage']:.1f}%)")
        
        # 检查比例一致性
        print(f"\n比例一致性检查:")
        unique_labels = np.unique(np.concatenate([train_labels, val_labels, test_labels]))
        
        for label in unique_labels:
            percentages = []
            for dataset_name in datasets.keys():
                if label in all_stats[dataset_name]:
                    percentages.append(all_stats[dataset_name][label]['percentage'])
                else:
                    percentages.append(0.0)
            
            std_dev = np.std(percentages)
            print(f"  {label}: 标准差 {std_dev:.2f}% (越小越均匀)")
            
            if std_dev > 5.0:  # 阈值可调整
                print(f"    警告: {label} 类别在各数据集中分布不够均匀!")
    
    def _save_split_dataset(self, images, masks, labels, file_path, split_name, metadata=None):
        """
        保存单个数据集分割
        
        Args:
            images: 图像数据
            masks: 掩码数据
            labels: 标签数据
            file_path: 保存路径
            split_name: 数据集名称
            metadata: 元数据
        """
        print(f"\n保存{split_name}数据集到: {file_path}")
        
        # 创建新的元数据
        split_metadata = {
            'split_type': split_name,
            'num_samples': len(images),
            'creation_time': pd.Timestamp.now().isoformat(),
            'split_ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            },
            'seed': self.seed
        }
        
        # 如果有原始元数据，合并
        if metadata is not None:
            original_metadata = metadata.item() if hasattr(metadata, 'item') else metadata
            if isinstance(original_metadata, dict):
                split_metadata.update(original_metadata)
        
        # 保存数据
        start_time = pd.Timestamp.now()
        np.savez(
            file_path,
            images=images,
            masks=masks,
            labels=labels,
            metadata=np.array([split_metadata], dtype=object)
        )
        
        save_time = pd.Timestamp.now() - start_time
        file_size = file_path.stat().st_size / (1024**2)
        
        # 统计标签分布
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        print(f"  保存完成! 用时: {save_time.total_seconds():.2f}秒")
        print(f"  文件大小: {file_size:.1f} MB")
        print(f"  样本数量: {len(images)}")
        print(f"  标签分布:", end=" ")
        for label, count in zip(unique_labels, counts):
            percentage = count / len(labels) * 100
            print(f"{label}={count}({percentage:.1f}%) ", end="")
        print()
    
    def load_split_info(self):
        """
        加载已划分数据集的信息
        
        Returns:
            dict: 包含各数据集信息的字典
        """
        split_files = {
            'train': self.train_npz,
            'val': self.val_npz,
            'test': self.test_npz
        }
        
        split_info = {}
        
        print("数据集分割信息:")
        print("=" * 50)
        
        for split_name, split_path in split_files.items():
            if split_path.exists():
                try:
                    data = np.load(split_path, allow_pickle=True)
                    
                    images = data['images']
                    labels = data['labels']
                    metadata = data.get('metadata', None)
                    
                    # 统计信息
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    file_size = split_path.stat().st_size / (1024**2)
                    
                    info = {
                        'path': split_path,
                        'samples': len(images),
                        'file_size_mb': file_size,
                        'label_distribution': dict(zip(unique_labels, counts)),
                        'metadata': metadata.item() if metadata is not None else None
                    }
                    
                    split_info[split_name] = info
                    
                    # 显示信息
                    print(f"{split_name.upper()}:")
                    print(f"  文件: {split_path}")
                    print(f"  样本数: {len(images)}")
                    print(f"  文件大小: {file_size:.1f} MB")
                    print(f"  标签分布: ", end="")
                    for label, count in zip(unique_labels, counts):
                        percentage = count / len(labels) * 100
                        print(f"{label}={count}({percentage:.1f}%) ", end="")
                    print()
                    
                    data.close()
                    
                except Exception as e:
                    print(f"{split_name.upper()}: 读取失败 - {e}")
                    split_info[split_name] = None
            else:
                print(f"{split_name.upper()}: 文件不存在 - {split_path}")
                split_info[split_name] = None
        
        print("=" * 50)
        return split_info

        
if __name__ == "__main__":

    # False and True to run data augmentation and split
    run_data_augmentation = False
    run_data_split = True

    if run_data_augmentation:
        aug_dataset = AugmentedDataset()
        images, masks, labels, filenames = aug_dataset.generate_augmented_data()
        aug_dataset.save_augmented_data(images, masks, labels, filenames, save_png=True)

    if run_data_split:
        splitter = DataSplit(
            train_ratio=0.7,   
            val_ratio=0.15,   
            test_ratio=0.15,      
        )
        
        split_datasets = splitter.split_and_save_dataset()
        
        split_info = splitter.load_split_info()
        
        batch_size = 16
        datasets_to_test = ['train', 'val', 'test']
        
        for dataset_type in datasets_to_test:
            try:
                print(f"\n测试 {dataset_type.upper()} 数据集:")
                print("-" * 20)
                
                # 标准采样
                print(f"标准采样:")
                ds_standard = ReadDataset(
                    dataset_type=dataset_type, 
                    batch_size=batch_size, 
                    auto_load=True, 
                    balanced_sampling=False
                )
                
                # 平衡采样
                print(f"\n平衡采样:")
                ds_balanced = ReadDataset(
                    dataset_type=dataset_type, 
                    batch_size=batch_size, 
                    auto_load=True, 
                    balanced_sampling=True
                )
                
                # 创建DataLoader (自动检测最佳num_workers设置)
                loader_standard = ds_standard.get_dataloader(shuffle=True)
                loader_balanced = ds_balanced.get_dataloader()
                
                # 测试批次分布
                def quick_test_distribution(loader, name, max_batches=2):
                    print(f"\n{name} - 前{max_batches}个批次测试:")
                    for batch_idx, (images_batch, masks_batch, labels_batch) in enumerate(loader):
                        if batch_idx >= max_batches:
                            break
                        
                        unique_labels, counts = np.unique(labels_batch, return_counts=True)
                        total_samples = len(labels_batch)
                        
                        print(f"  批次{batch_idx+1}: ", end="")
                        for label, count in zip(unique_labels, counts):
                            percentage = count / total_samples * 100
                            print(f"{label}={count}({percentage:.1f}%) ", end="")
                        print()
                
                quick_test_distribution(loader_standard, "标准采样")
                quick_test_distribution(loader_balanced, "平衡采样")
                
                # 关闭数据集
                ds_standard.close()
                ds_balanced.close()
                
            except FileNotFoundError:
                print(f"  {dataset_type.upper()} 数据集文件不存在，请先执行数据集划分")
            except Exception as e:
                print(f"  测试 {dataset_type.upper()} 数据集时出错: {e}")
        