# Fluorescence microscopy segmentation
This is an exploration of the different machine learning techniques to
segment images of fluorescence microscopy. The goal of this endeavor is to
find a suitable approach that is highly accurate while also feasible with the
resources typically available to a lab. All of the protocols below use the
FluoCells dataset (https://amsacta.unibo.it/id/eprint/6706/), a set of 283
1600x1200 images of mice brain slices and their corresponding masks.

## Supervised learning
### U-Net
The U-Net architecture is a fully convolutional neural network created for
image segmentation and named after the shape of the network consisting of a
contractive and expansive path in the network. Current goals include:
  - Improving loss metric for increased penalties between two cells that are
  very close/touching
  - Creating methodology for making images appropriate size before training/
  testing using image patches rather than resizing the image. This can be
  achieved through either:
    - Restricting cropping to the bounds of the image. 
    - Extend cropping outside of the bounds of the image and mirror portions
    of the image as described in https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

## Unsupervised learning
### K mean clustering
K means clustering is an iterative process of grouping a collection of
observations into K groups. This operates under the assumption that there
are at least K groups present within the data. 

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
