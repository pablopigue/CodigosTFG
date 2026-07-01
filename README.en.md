> 🌐 [Versión en español](README.md)

# TFG Code

Code for the Final Degree Project (TFG) on GAN architectures. Six
noise-to-image generation models are compared (Vanilla GAN, DCGAN, DCGAN
with label smoothing, WGAN with weight clipping, convolutional WGAN, and
WGAN-GP) across four datasets (MNIST, FashionMNIST, SVHN, and CelebA),
and an image-to-image translation model (CycleGAN) is trained on
apple2orange.

For the noise-to-image generation models, FID and Inception Score are
computed against the same fixed subset of 10,000 real images (seed 42),
so that the values are comparable across models and across runs of the
same model. For CycleGAN, only FID is computed, since IS does not make
sense for this architecture.


## Repository structure

```
CodigosTFG/
├── CodigosRefactorizados/            Modularized scripts for training the GANs.
├── ResultadosCodigosGeneralizacion/  CSVs and results produced by the training runs.
├── PlotsArquitecturas/               Code used to create the architecture figures.
│
├── comparar_losses.py                Compares G and D/C loss curves across models.
├── estadisticas_tabla.py             Tables with final FID/IS per model and dataset.
├── generar_comparativas_general.py   Global FID/IS comparisons across all models.
│
├── Plot_dataset_samples.py           Sample grids for each dataset.
├── Plot_grid_mejor_fid.py            Sample grids at the best-FID checkpoint.
├── Plot_losses_con_zoom_mnist.py     Loss curves zoomed in to show MNIST instabilities.
│
└── README.md
```


## Datasets

| Dataset       | Resolution | Channels | Epochs | Runs |
|---------------|------------|----------|--------|------|
| MNIST         | 32×32      | 1        | 80     | 10   |
| FashionMNIST  | 32×32      | 1        | 80     | 10   |
| SVHN          | 32×32      | 3        | 150    | 10   |
| CelebA        | 64×64      | 3        | 40     | 5    |
| apple2orange  | 256×256    | 3        | 200    | 1    |

MNIST, FashionMNIST and SVHN are resized to 32×32 and downloaded with
`torchvision`. CelebA is resized to 64×64 and loaded via `ImageFolder`
from a local directory. apple2orange (used only by CycleGAN) is loaded
from disk with a custom `Dataset`: during training the images are
resized to 286×286 and randomly cropped to 256×256, and during
evaluation they are resized directly to 256×256, following the protocol
of Zhu et al. (2017).


## CodigosRefactorizados/

The scripts for the noise-to-image generation models and the cycleGAN model.

### `utils.py`

It is organized into five blocks:

- A. Data loading: `build_transform`, `load_dataset` (MNIST,
  FashionMNIST, SVHN via `torchvision`; CelebA via `ImageFolder`).
- B. Fixed evaluation subset: `build_fixed_eval_set` samples the real
  images used for FID/IS with seed 42, and then resets the seed so that
  the training runs remain independent.
- C. Metrics: `compute_fid_is` computes FID and IS with `torchmetrics`.
  It works both with MLP generators (flat output) and convolutional ones
  (4D output) via the `flatten_output` flag.
- D. Initialization and saving: `weights_init_dcgan` (init from Radford
  et al.), `save_sample_images`, `save_plot`.
- E. Multi-run: `make_experiment_dirs`, `save_run_artifacts` (weights,
  CSV and plots of the first run) and `aggregate_runs` (average over the
  N runs with standard-deviation bands).

### Training scripts

The scripts always follow the same outline: configuration, data loading,
fixed subset, architecture definition, multi-run loop and
post-processing.

#### General models (MNIST / FashionMNIST / SVHN)

Any of the three datasets is selected by changing `DATASET_NAME` at the
top of the file.

| Script                      | Model        | Architecture | Loss                                       | Optimizer            |
|-----------------------------|--------------|--------------|--------------------------------------------|----------------------|
| `tfg_vanilla_general.py`    | Vanilla GAN  | MLP          | BCE                                        | Adam (β=0.5, 0.999)  |
| `tfg_dcgan_general.py`      | DCGAN        | CNN          | BCE                                        | Adam (β=0.5, 0.999)  |
| `tfg_dcgan_label.py`        | DCGAN + LS   | CNN          | BCE with label smoothing (0.9)             | Adam (β=0.5, 0.999)  |
| `tfg_wgan_general.py`       | WGAN         | MLP          | Wasserstein with weight clipping 0.01      | RMSprop              |
| `tfg_wgan_conv_general.py`  | WGAN-Conv    | CNN          | Wasserstein with weight clipping 0.01      | RMSprop              |
| `tfg_wgangp_general.py`     | WGAN-GP      | CNN          | Wasserstein with gradient penalty (λ=10)   | Adam (β=0.0, 0.9)    |

The hyperparameters follow the values proposed in the original papers
(Goodfellow et al. 2014, Radford et al. 2015, Arjovsky et al. 2017,
Gulrajani et al. 2017).

WGAN-Conv deserves a separate note: it uses the same convolutional
critic as WGAN-GP but is trained with weight clipping instead of
gradient penalty. It serves to separate the effect of the architecture
(MLP vs CNN) from the effect of the regularization method (clipping vs
GP) in the WGAN results.

#### Models for CelebA

Adaptations to 64×64 and to the CelebA dataset.

| Script                          | Model                     |
|---------------------------------|---------------------------|
| `tfg_vanilla_celeba.py`         | Vanilla GAN               |
| `tfg_dcgan_celeba.py`           | DCGAN                     |
| `tfg_dcgan_label_celeba.py`     | DCGAN with label smoothing |
| `tfg_wgan_celeba.py`            | WGAN                      |
| `tfg_wgan_conv_celeba.py`       | Convolutional WGAN        |
| `tfg_wgangp_celeba.py`          | WGAN-GP                   |

#### CycleGAN

`tfg_cyclegan.py` implements the version of Zhu et al. (2017): a 9-block
ResNet generator, a 70×70 PatchGAN discriminator, adversarial loss plus
cycle loss (λ=10) plus identity loss (λ=5). Adam optimizer with a linear
learning-rate decay starting at epoch 100 (out of 200 total). It is
trained on apple2orange.

#### Timing benchmark

`benchmark_tiempos.py` measures how long one training epoch takes for
each model-by-dataset combination. It runs 1 warmup epoch that is not
timed and 5 measured epochs, with `torch.cuda.synchronize()` before and
after each one. It does not compute FID/IS or save images, so the
measurement reflects only the cost of adversarial training.

`analyze_benchmark.py`, located in /ResultadosCodigosGeneralizacion/Tiempos,
takes the previous results and generates the LaTeX table and the plots
for time per epoch, time per batch and scaling with resolution.

## Logged metrics

Each training run generates a `metrics_all_runs.csv` with the columns:

| Column      | Description                                                 |
|-------------|-------------------------------------------------------------|
| `epoch`     | Epoch number                                                |
| `run`       | Run identifier (1…N)                                        |
| `loss_g`    | Generator loss                                              |
| `loss_d`    | Discriminator loss (BCE models)                             |
| `loss_c`    | Critic loss (Wasserstein models)                            |
| `gp`        | Gradient penalty term (WGAN-GP only)                        |
| `fid`       | FID, computed every `CALC_METRICS_FREQ` epochs              |
| `is_mean`   | Inception Score, mean                                       |
| `is_std`    | Inception Score, standard deviation                         |

For CycleGAN, IS is not saved (it is not computed) and the following are
also saved: `loss_GAN`, `loss_cycle`, `loss_identity`, `loss_D_A`,
`loss_D_B`, `fid_AB`, `fid_BA` and `lr`.


## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 with CUDA support
- torchvision, torchmetrics
- pandas, numpy, matplotlib, Pillow

```bash
pip install torch torchvision torchmetrics pandas numpy matplotlib pillow
```

For CelebA, download the dataset and place it in a directory with a
structure compatible with `ImageFolder` (the images inside any
subfolder; the exact path is set in `DATA_DIR` within each script). For
apple2orange (CycleGAN), the expected structure is `trainA/`, `trainB/`,
`testA/`, `testB/`.


## Usage

Each script is run independently:

```bash
cd CodigosRefactorizados
python tfg_dcgan_general.py          # DCGAN on the dataset set by DATASET_NAME
python tfg_wgan_conv_general.py      # WGAN-Conv on the dataset set by DATASET_NAME
python tfg_wgangp_celeba.py          # WGAN-GP on CelebA
python tfg_cyclegan.py               # CycleGAN on apple2orange
python benchmark_tiempos.py          # Training times
python analyze_benchmark.py          # Benchmark tables and plots
```

Results are written to the directory given by `EXPERIMENT_DIR` in each
script, with the structure:

```
experimento/
├── images/    Grids of images generated during training.
├── models/    Final weights (G and D/C) of the first run.
├── logs/      CSVs with per-epoch and averaged metrics.
└── plots/     Loss, FID and IS curves (individual and averaged).
```

## Note on the plotting code

The images included in the TFG are taken from those scripts, but some
were edited manually afterwards, since it was faster to change small
details by hand than in code.

## References

- Goodfellow, I. et al. (2014). Generative Adversarial Networks.
- Radford, A., Metz, L., Chintala, S. (2015). Unsupervised Representation
  Learning with Deep Convolutional GANs. arXiv:1511.06434.
- Arjovsky, M., Chintala, S., Bottou, L. (2017). Wasserstein GAN.
  arXiv:1701.07875.
- Gulrajani, I. et al. (2017). Improved Training of Wasserstein GANs.
  arXiv:1704.00028.
- Zhu, J.-Y. et al. (2017). Unpaired Image-to-Image Translation using
  Cycle-Consistent Adversarial Networks. arXiv:1703.10593.
- Salimans, T. et al. (2016). Improved Techniques for Training GANs.
  arXiv:1606.03498.
