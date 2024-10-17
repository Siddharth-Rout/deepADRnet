
# Advection Augmented Convolutional Neural Networks (ADR-CNN)

**Authors**: Niloufar Zakariaei, Siddharth Rout, Eldad Haber, Moshe Eliasof

**Conference**: 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

## Abstract

Many problems in the physical sciences involve the prediction of space-time sequences, such as weather forecasting, disease propagation, and video prediction. Traditional approaches using Convolutional Neural Networks (CNNs) combined with time-prediction mechanisms often struggle with long-range information propagation and lack explainability. To address these issues, we propose a novel architecture that augments CNNs with advection using a semi-Lagrangian push operator. This operator enables the non-local transformation of information, which is particularly beneficial in problems requiring rapid information transport across an image. We combine this operator with reaction and diffusion components to form a network that mirrors the Reaction-Advection-Diffusion (ADR) equation. We demonstrate the superior performance of this approach on various spatio-temporal datasets.

---

## Key Contributions

1. **Advection-Augmented Architecture**: We introduce a physically inspired network that augments CNNs with a semi-Lagrangian push operator to model advection, allowing for non-local information transport.
   
2. **ADR Process**: The network mimics the Reaction-Advection-Diffusion equation to solve spatio-temporal prediction problems, offering explainability by linking neural operations to known physical processes.

3. **Performance**: Our ADRNet outperforms traditional CNN architectures in scenarios requiring the transport of features over long distances, such as weather prediction and traffic flow analysis.

## Model Overview

- **Advection Component**: Moves information across an image using the semi-Lagrangian method, overcoming the limitations of local convolutional operations.
- **Diffusion and Reaction Components**: Modeled as standard CNN layers, these components perform smoothing and local feature updates.
- **Operator Splitting**: Inspired by differential equation solvers, our model divides the ADR process into three steps—advection, diffusion, and reaction—to improve efficiency and stability.

## Datasets

We evaluate the ADR-CNN on several spatio-temporal datasets:

- **Shallow Water Equation (SWE)**: Derived from Navier-Stokes equations.
- **CloudCast**: Satellite imagery for weather prediction.
- **Moving MNIST**: Synthetic video prediction dataset.
- **KITTI**: Benchmark dataset for video prediction in autonomous driving.

## Citation

If you use this work in your research, please cite:

```
@inproceedings{ADR-CNN2024,
  title={Advection Augmented Convolutional Neural Networks},
  author={Niloufar Zakariaei, Siddharth Rout, Eldad Haber, Moshe Eliasof},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```
