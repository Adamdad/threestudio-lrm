# threestudio-lrm
The threestudio-lrm is an extension for threestudio, integrating the Large Reconstruction Model (LRM) for advanced 3D reconstruction tasks. This model is based on the framework detailed at [https://yiconghong.me/LRM/](https://yiconghong.me/LRM/). We have adapted the open-source implementation from [OpenLRMn](https://github.com/3DTopia/OpenLRM). Currently, it is only used for initialization of Gaussian Splatting.

## Installation
```
cd custom
git clone https://github.com/Adamdad/threestudio-lrm

# install openLRM
cd threestudio-lrm
git clone https://github.com/3DTopia/OpenLRM.git
cd OpenLRM
pip install -r requirements.txt
```

## Examples
Please see [threestudio-3dgs](https://github.com/DSaurus/threestudio-3dgs#load-from-ply) for more details.

## Citation
If you use threestudio-lrm in your research, please cite the following paper:
```
@article{hong2023lrm,
  title={Lrm: Large reconstruction model for single image to 3d},
  author={Hong, Yicong and Zhang, Kai and Gu, Jiuxiang and Bi, Sai and Zhou, Yang and Liu, Difan and Liu, Feng and Sunkavalli, Kalyan and Bui, Trung and Tan, Hao},
  journal={arXiv preprint arXiv:2311.04400},
  year={2023}
}
```
