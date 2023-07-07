# GyroFlow+: Gyroscope-Guided Unsupervised Deep Homography and Optical Flow Learning
[Haipeng Li](https://lhaippp.github.io/), [Kunming Luo](https://coolbeam.github.io/index.html), [Bing Zeng](https://scholar.google.com.hk/citations?user=4y0QncgAAAAJ&hl=zh-CN), [Shuaicheng Liu](http://www.liushuaicheng.org/)

## GHOF Dataset
- Benchmark consists of GHOF-Clean and GHOF-Final is available at [GoogleDrive]( https://drive.google.com/drive/folders/1Un1rK777AEuz1tT3MJ7OTgZt2PHtIwRW?usp=sharing)
- Download the two BMKs and put to root path

## Test Demo
`python test.py --model_dir experiments/ --restore_file experiments/step_418600_homo.pth`

## Citation

```
@inproceedings{li2021gyroflow,
  title={Gyroflow: gyroscope-guided unsupervised optical flow learning},
  author={Li, Haipeng and Luo, Kunming and Liu, Shuaicheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12869--12878},
  year={2021}
}

@article{li2023gyroflow+,
  title={GyroFlow+: Gyroscope-Guided Unsupervised Deep Homography and Optical Flow Learning},
  author={Li, Haipeng and Luo, Kunming and Zeng, Bing and Liu, Shuaicheng},
  journal={arXiv preprint arXiv:2301.10018},
  year={2023}
}
```
