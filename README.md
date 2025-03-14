# Fluorescence microscopy segmentation
This is an exploration of the different machine learning techniques to
segment images of fluorescence microscopy. All of the protocols below use the
FluoCells dataset (https://amsacta.unibo.it/id/eprint/6706/), a set of 283
1600x1200 images of mice brain slices and their corresponding masks.

## Supervised learning
### U-Net
The U-Net architecture is a fully convolutional neural network created for
image segmentation and named after the shape of the network consisting of a
contractive and expansive path in the network.

## Unsupervised learning
### K mean clustering


### W-Net
The W-Net was inspired by the U-Net and (as the name entails) is two
consecutive U-Net networks, one functioning as an encoder and the other as a
decoder. 

## Other
### Otsu's method
Although not technically within the realm of machine learning, Otsu's method
is a process used in automatic image segmentation. There is a one and two
dimensional variant, with the two dimensional variant being more adept at
segmenting noisy images. Otsu's method operates on the assumption that there
are two classes present in the image being segmented. 
