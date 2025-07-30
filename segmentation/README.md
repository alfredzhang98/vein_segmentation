Learning pytorch form this [link](https://github.com/lucidrains)

Learning unet pytorch [link](https://github.com/milesial/Pytorch-UNet)

# Introduction of Unet Segmentation

Dataset and trained model files: [Google Drive](https://drive.google.com/drive/folders/1K7I51N4xtgEfaKAJtppvmrK9fP8AfKHu?usp=drive_link)

## Data Collection

`DataInfo.py` contains the configuration for data collection, including the path of the whole dataset, the path of the images, the path of the masks, the image size, and the augmentation types. You can modify these parameters according to your needs.

`dataCollection.py` is used to collect B mode ultrasound data from the Clarius ultrasound device. Recommended to connect the device WIFI directly on your computer. The data will be saved in the `data` directory. There is a `csv` file that contains the png file names and their information and their related path.

## labeling the Data
`dataLabel.py` is used to label the data collected in the previous step. It uses `brushes` to label the data. The labeled data will be saved in the `data` directory. The `csv` file will be updated with the labeled data information.

## Data Augmentation
`dataPrepare.py` is used to augment, splict, and prepare the Dataset for training and validation. 

## Training and validation 
`train.py` is used to train the Unet model. It uses the `torch` library to train the model. The model will be saved in the `outputs` directory. The training process will be logged in the `wandb` online platform. You can see the training process in the `wandb` dashboard. (You need to login to the `wandb` platform first with your own account.)

## Testing
`test.py` is used to test the Unet model.

## Prediction
`predict.py` is used to predict the segmentation of the ultrasound images.

# Clarius API
`cast`  dictionary, `pyclariuscast.pyd` (windows), `pyclariuscast.so` (linux), `cast.dll` and `cast.lib` are the main files you need to run Clarius API with Python.

For linux it should install 
```
conda install -c conda-forge gcc_linux-64 gxx_linux-64 libstdcxx-ng=12 libffi=3.4.2 glib
```

In linux if `qt.qpa.plugin: Could not load the Qt platform plugin "wayland" in "" even though it was found.` appears:

```
// Run this command in terminal before running the python script
export QT_QPA_PLATFORM=xcb
```