Learning pytorch form this [link](https://github.com/lucidrains)

Learning unet pytorch [link](https://github.com/milesial/Pytorch-UNet)

# Introduction of Unet Segmentation

Dataset and trained model files: [Google Drive](https://drive.google.com/drive/folders/1K7I51N4xtgEfaKAJtppvmrK9fP8AfKHu?usp=drive_link)

## Data Collection

`0collectiondata.py` is used to collect B mode ultrasound data from the Clarius ultrasound device. Recommended to connect the device WIFI directly on your computer. The data will be saved in the `data` directory. There is a `csv` file that contains the png file names and their information and their related path.

## labeling the Data
`1labeldata.py` is used to label the data collected in the previous step. It uses `brushes` to label the data. The labeled data will be saved in the `data` directory. The `csv` file will be updated with the labeled data information.

## Data Augmentation
`2augdata.py` is used to augment the data collected in the previous step. It uses `cv2` to augment the data. The augmented data will be saved in the `data` directory. The `csv` file will be updated with the augmented data information.

## Data Preparation
`dataSplit.ipynb` is used to prepare the data for training. It splits the data into training, validation, and test sets. The `train.csv`, `val.csv`, and `test.csv` files will be created in the `outputs` directory. 

`dataSizeMap.py` is a class not run itself.
It will be run in the `train.py` file to get the data in tensor format.
It includes the `__getitem__` method to get the data in tensor format and the `__len__` method to get the length of the data.
It also includes the padding method to pad the data to the same size.
Now the size is 576x544, which could be devide by 32, because it is easy to downsample the data in the Unet model.

## Training and validation 
`train.py` is used to train the Unet model. It uses the `torch` library to train the model. The model will be saved in the `outputs` directory. The training process will be logged in the `wandb` online platform. You can see the training process in the `wandb` dashboard. (You need to login to the `wandb` platform first with your own account.)

## Testing
`test.py` is used to test the Unet model.

# Clarius API
`cast`  dictionary, `pyclariuscast.pyd`, `cast.dll` and `cast.lib` are the main files you need to run Clarius API with Python.