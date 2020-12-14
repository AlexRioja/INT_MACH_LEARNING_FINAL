# INT_MACH_LEARNING Final Proyect

This is the repository for the final proyect of INT_MACH_LEARNING subject.

## Models used
GoogleNet and VGG16, trained on ImageNet

## File explanation

The files corresponding with the code to run on CPU are:

```bash
inception_V3.py
```
```bash
vgg_16.py
```
The files to run the models in the NCS2 device (MYRIAD) are:
```bash
inception_V3_NCS2.py
```
```bash
vgg_16_NCS2.py
```
The files to freeze the models, and transform them to .pb, (to be compiled by mo.py from OpenVINO later on) are:
```bash
keras2ncs.py-->GoogLeNet custom trained to .pb
```
```bash
keras2ncs_vgg.py-->VGG16 pretrained to .pb
```
## Usage
Simply run the file you want. Press 'q' to close the windows

## Dependencies
OpenCV,
Keras,
Tensorflow (2 for running inception_V3.py and vgg_16.py and 1 for the others)

## License
[MIT](https://choosealicense.com/licenses/mit/)
