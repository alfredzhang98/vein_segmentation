# Clarius API
`cast`  dictionary, `pyclariuscast.pyd`, `cast.dll` and `cast.lib` are the main files you need to run Clarius API with Python.

`DICOMserver.py` is to create a DICOM server on your computer, it could only be accessed when the device is connected to a Wi-Fi network(not be a hotspot device). It will save the DICOM files to the `dicom` directory when you finish the scan and report it end on the Clarius app.
`pycaster.py` is a simple example to show how to use Clarius API, it should run with arguments like python pycaster.py --ip
`pysidecaster_fps.py` and `pysidecaster.py` is use pyside6 to create a simple GUI for Clarius API.
`pyimu.py` is a simple gui example to show how to use the IMU data from Clarius API. BUGS yaw is not accurate.


