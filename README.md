# MAC Network Augmented With Multiple Memories
Memory, Attention and Composition (MAC) Network for CLEVR from Compositional Attention Networks for Machine Reasoning (https://arxiv.org/abs/1803.03067) implemented in PyTorch

Requirements:
* Python 3.6
* PyTorch 1.*
* torch-vision
* Pillow
* nltk
* tqdm

To train:

1. Download and extract CLEVR v1.0 dataset from http://cs.stanford.edu/people/jcjohns/clevr/
2. Preprocessing question data and extracting image features using ResNet 101
```
python preprocess.py [CLEVR directory]
python image_feature.py [CLEVR directory]
```
!CAUTION! the size of file created by image_feature.py is very large! (~70 GiB) You may use hdf5 compression, but it will slow down feature extraction.

3. Run train.py
```
python train.py [CLEVR directory]
```

Run `python train.py -h` for more options

## Differences from MAC:
Pending
