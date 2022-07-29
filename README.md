<div align="center">

  # SPECTRE: Visual Speech-Aware Perceptual 3D Facial Expression Reconstruction from Videos

[![Paper](https://img.shields.io/badge/arXiv-2207.11094-brightgreen)](https://arxiv.org/abs/2207.11094)
[![Project WebPage](https://img.shields.io/badge/Project-webpage-blue)](https://filby89.github.io/spectre/)

</div>

<p align="center"> 
<img src="cover.png">
</p>
<p align="center"> Our method performs visual-speech aware 3D reconstruction so that speech perception from the original footage is preserved in the reconstructed talking head. On the left we include the word/phrase being said for each example. <p align="center">

This is the official Pytorch implementation of the paper:
  
```
Visual Speech-Aware Perceptual 3D Facial Expression Reconstruction from Videos
Panagiotis P. Filntisis, George Retsinas, Foivos Paraperas-Papantoniou, Athanasios Katsamanis, Anastasios Roussos, and Petros Maragos
arXiv 2022
```



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
python demo.py --input samples/LRS3/0Fi83BHQsMA_00002.mp4 --audio
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

## Citation
If your research benefits from this repository, consider citing the following:

```
@misc{filntisis2022visual,
  title = {Visual Speech-Aware Perceptual 3D Facial Expression Reconstruction from Videos},
  author = {Filntisis, Panagiotis P. and Retsinas, George and Paraperas-Papantoniou, Foivos and Katsamanis, Athanasios and Roussos, Anastasios and Maragos, Petros},
  publisher = {arXiv},
  year = {2022},
}
```
  
  
