# Tri-MipRF nerfstudio integration

This repository provides code to integrate the [Tri-MipRF](https://wbhu.github.io/projects/Tri-MipRF) into [nerfstudio](https://docs.nerf.studio/en/latest/index.html).

<div align='center'>
    <img src="https://wbhu.github.io/projects/Tri-MipRF/img/overview.jpg" height="200px"/>
</div>

It provides an alternative way to use `Tri-Mip Encoding` in addition to the [official repository](https://github.com/wbhu/Tri-MipRF), which allows access to nerfstudio's in-browser viewer and additional training capabilities. Beware that some details about the training procedure differ from the official repository.

## **Installation**

1. [Install nerfstudio](https://docs.nerf.studio/en/latest/quickstart/installation.html). This is `pip install nerfstudio`, but there are a few dependencies (e.g. `torch`, `tinycudann`) which may require further steps, so make sure to check their installation guide!
2. Install the trimiprf nerfstudio integration (this repository): `pip install trimiprf-nerfstudio`

### Running Model

```bash
ns-train trimiprf --data <data-folder>
```


## Roadmap

Expected future updates to this repository:

 - [ ] Support multi-scale blender dataset



This file is modified base on  [kplanes_nerfstudio](https://github.com/Giodiro/kplanes_nerfstudio).