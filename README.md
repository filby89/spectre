<div align="center">
# SPECTRE: Visual Speech-Aware Perceptual 3D Facial Expression Reconstruction from Videos


<p align="center"> 
<img src="cover.png">
</p>
<p align="center"> Our method performs visual-speech aware 3D reconstruction so that speech perception from the original footage is preserved in the reconstructed talking head. On the left we include the word/phrase being said for each example. <p align="center">
</div>

This is the official Pytorch implementation of SPECTRE. 

## Installation
You need to have installed a working version of Pytorch with Python 3.6 or higher and Pytorch 3D.

Clone the repo and its submodules:
  ```bash
  git clone --recurse-submodules -j4 https://github.com/filby89/spectre
  cd spectre
  ```  

Install the face_alignment and face_detection packages:
```bash
cd external/face_alignment
pip install -e .
cd ../face_detection
pip install -e .
```

Download the FLAME model and the pretrained SPECTRE model:
```bash
bash quick_install.sh
```



[//]: # (Create a new conda environment and install the requirements:)

[//]: # (```bash)

[//]: # (conda create -n "spectre" python=3.8)

[//]: # (pip install -r requirements.txt)

[//]: # (```)

## Demo
Samples are included in ``samples`` folder. You can run the demo by running 

```bash
python demo.py --input samples/MEAD/M003_level_1_happy_024.mp4 --audio
```

The audio flag extracts audio from the input video and puts it in the output shape video for visualization purposes. More options and samples will be available soon.

## Training and Testing
Training code will be released shortly in the following week after some refactoring for easier usage.


## Acknowledgements
This repo is has been heavily based on the original implementation of [DECA](https://github.com/YadiraF/DECA/). We also acknowledge the following 
repositories which we have benefited greatly from as well:


- [EMOCA](https://github.com/radekd91/emoca)
- [face_alignment](https://github.com/hhj1897/face_alignment)
- [face_detection](https://github.com/hhj1897/face_detection)
- [Visual_Speech_Recognition_for_Multiple_Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)
