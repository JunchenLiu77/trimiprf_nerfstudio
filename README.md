# Tri-MipRF nerfstudio integration

This repository provides code to integrate the [Tri-MipRF](https://wbhu.github.io/projects/Tri-MipRF) into [nerfstudio](https://docs.nerf.studio/en/latest/index.html).

<div align='center'>
    <img src="https://wbhu.github.io/projects/Tri-MipRF/img/overview.jpg" height="200px"/>
</div>

It provides an alternative way to use `Tri-Mip Encoding` in addition to the [official repository](https://github.com/wbhu/Tri-MipRF), which allows access to nerfstudio's in-browser viewer and additional training capabilities. Beware that some details about the training procedure differ from the official repository.

## Installation

1. Install [nerfstudio](https://docs.nerf.studio/en/latest/quickstart/installation.html). This is `pip install nerfstudio`, but there are a few dependencies (e.g. `torch`, `tinycudann`) which may require further steps, so make sure to check their installation guide!
2. Install [nvdiffrast](https://nvlabs.github.io/nvdiffrast/).
3. Install the trimiprf nerfstudio integration (this repository): `pip install .`

## Running Model

```bash
ns-train trimiprf --data <data-folder>
```

## Benchmarks

> The observed decrease in performance, when compared to the official implementation, is primarily due to the choice of optimizer. However, the optimizer utilized in the official implementation does not enhance performance as expected. PRs that propose modifications to the optimizer or suggest alternative methods to improve performance are highly welcomed.

**Single Scale Synthetic NeRF**

|           | chair | drums | ficus | hotdog | lego  | materials | mic   | ship  |
| --------- | ----- | ----- | ----- | ------ | ----- | --------- | ----- | ----- |
| **PSNR**  | 34.69 | 24.51 | 30.90 | 35.76  | 34.38 | 27.56     | 35.12 | 18.47 |
| **SSIM**  | 0.980 | 0.913 | 0.967 | 0.974  | 0.975 | 0.917     | 0.987 | 0.620 |
| **LPIPS** | 0.014 | 0.083 | 0.032 | 0.026  | 0.012 | 0.073     | 0.011 | 0.465 |

## Roadmap

Expected future updates to this repository:

 - [ ] Support multi-scale blender dataset



This file is modified base on  [kplanes_nerfstudio](https://github.com/Giodiro/kplanes_nerfstudio).