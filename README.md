> 🌐 [English version](README.en.md)
# Códigos TFG

Código del Trabajo de Fin de Grado sobre arquitecturas GAN. Se comparan
seis modelos de generación a partir de ruido (Vanilla GAN, DCGAN, DCGAN
con label smoothing, WGAN con weight clipping, WGAN convolucional y
WGAN-GP) sobre cuatro datasets (MNIST, FashionMNIST, SVHN y CelebA), y
se realiza un modelo de traducción imagen a imagen (CycleGAN) 
sobre apple2orange.

En los modelos de generación a partir de ruido se calculan FID e 
Inception Score contra un mismo subconjunto fijo de 10 000 
imágenes reales (semilla 42), de forma que los valores son 
comparables entre modelos y entre ejecuciones de un
mismo modelo. En caso de Cyclegan solo se calcula FID pues
IS no tiene sentido en esta arquitectura.


## Estructura del repositorio

```
CodigosTFG/
├── CodigosConLosQueSeEntreno/        Scripts originales con los que se entrenó MNIST, FashionMNIST y SVHN.
├── CodigosConQueSeEntrenoCELEBA/     Scripts originales con los que se entrenó CelebA.
├── CodigosRefactorizados/            Versión modularizada de los anteriores (ver más abajo).
├── ResultadosCodigosGeneralizacion/  CSVs y resultados generados por los entrenamientos.
├── PlotsArquitecturas/               Códigos usados para crear imágenes de las arquitecturas
│
├── comparar_losses.py                Compara curvas de pérdida G y D/C entre modelos.
├── estadisticas_tabla.py             Tablas con FID/IS finales por modelo y dataset.
├── generar_comparativas_general.py   Comparativas globales FID/IS entre todos los modelos.
│
├── Plot_dataset_samples.py           Cuadrículas de muestra de cada dataset.
├── Plot_grid_mejor_fid.py            Cuadrículas en el checkpoint de mejor FID.
├── Plot_losses_con_zoom_mnist.py     Curvas de pérdida con zoom para ver inestabilidades en MNIST.
│
└── README.md
```


## Datasets

| Dataset       | Resolución | Canales | Épocas | Ejecuciones |
|---------------|------------|---------|--------|-------------|
| MNIST         | 32×32      | 1       | 80     | 10          |
| FashionMNIST  | 32×32      | 1       | 80     | 10          |
| SVHN          | 32×32      | 3       | 150    | 10          |
| CelebA        | 64×64      | 3       | 40     | 5           |
| apple2orange  | 256×256    | 3       | 200    | 1           |

MNIST, FashionMNIST y SVHN se redimensionan a 32×32 y se descargan con
`torchvision`. CelebA se redimensiona a 64×64 y se carga vía
`ImageFolder` desde un directorio local. apple2orange (usado solo por
CycleGAN) se carga desde disco con un `Dataset` propio: en entrenamiento
las imágenes se redimensionan a 286×286 y se recorta aleatoriamente a
256×256, y en evaluación se redimensionan directamente a 256×256,
siguiendo el protocolo de Zhu et al. (2017).


## CodigosRefactorizados/

Los scripts originales para los modelos de generación a partir de ruido
tenían mucho código duplicado: carga de datos, construcción del subconjunto 
de evaluación, cálculo de FID/IS, guardado de imágenes, generación de 
gráficas y agregación multi-run aparecían prácticamente idénticos en 
cada fichero. En esta versión todo eso se ha movido a `utils.py`, 
y los scripts de cada modelo se quedan únicamente con su arquitectura, 
su función de pérdida y su bucle de entrenamiento.

### `utils.py`

Está organizado en cinco bloques:

- A. Carga de datos: `build_transform`, `load_dataset` (MNIST,
  FashionMNIST, SVHN vía `torchvision`; CelebA vía `ImageFolder`).
- B. Subconjunto fijo de evaluación: `build_fixed_eval_set` muestrea con
  semilla 42 las imágenes reales que se usarán para FID/IS, y después
  resetea la semilla para que los runs de entrenamiento sigan siendo
  independientes.
- C. Métricas: `compute_fid_is` calcula FID e IS con `torchmetrics`.
  Funciona tanto con generadores MLP (salida plana) como con
  convolucionales (salida 4D) mediante el flag `flatten_output`.
- D. Inicialización y guardado: `weights_init_dcgan` (init de Radford et
  al.), `save_sample_images`, `save_plot`.
- E. Multi-run: `make_experiment_dirs`, `save_run_artifacts` (pesos, CSV
  y gráficas del primer run) y `aggregate_runs` (promedio de las N
  ejecuciones con bandas de desviación estándar).

### Scripts de entrenamiento

Los scripts siguen siempre el mismo esquema: configuración, carga de
datos, subconjunto fijo, definición de la arquitectura, bucle multi-run
y postprocesado.

#### Modelos generales (MNIST / FashionMNIST / SVHN)

Cualquiera de los tres datasets se selecciona cambiando `DATASET_NAME`
al principio del fichero.

| Script                      | Modelo       | Arquitectura | Pérdida                                  | Optimizador          |
|-----------------------------|--------------|--------------|------------------------------------------|----------------------|
| `tfg_vanilla_general.py`    | Vanilla GAN  | MLP          | BCE                                      | Adam (β=0.5, 0.999)  |
| `tfg_dcgan_general.py`      | DCGAN        | CNN          | BCE                                      | Adam (β=0.5, 0.999)  |
| `tfg_dcgan_label.py`        | DCGAN + LS   | CNN          | BCE con label smoothing (0.9)            | Adam (β=0.5, 0.999)  |
| `tfg_wgan_general.py`       | WGAN         | MLP          | Wasserstein con weight clipping 0.01     | RMSprop              |
| `tfg_wgan_conv_general.py`  | WGAN-Conv    | CNN          | Wasserstein con weight clipping 0.01     | RMSprop              |
| `tfg_wgangp_general.py`     | WGAN-GP      | CNN          | Wasserstein con gradient penalty (λ=10)  | Adam (β=0.0, 0.9)    |

Los hiperparámetros siguen los valores propuestos en los papers
originales (Goodfellow et al. 2014, Radford et al. 2015, Arjovsky et al.
2017, Gulrajani et al. 2017).

WGAN-Conv merece una nota aparte: usa el mismo crítico convolucional que
WGAN-GP pero se entrena con weight clipping en vez de gradient penalty.
Sirve para separar el efecto de la arquitectura (MLP vs CNN) del efecto
del método de regularización (clipping vs GP) en los resultados de WGAN.

#### Modelos para CelebA

Adaptaciones a 64×64 y al dataset CelebA.

| Script                          | Modelo                    |
|---------------------------------|---------------------------|
| `tfg_vanilla_celeba.py`         | Vanilla GAN               |
| `tfg_dcgan_celeba.py`           | DCGAN                     |
| `tfg_dcgan_label_celeba.py`     | DCGAN con label smoothing |
| `tfg_wgan_celeba.py`            | WGAN                      |
| `tfg_wgan_conv_celeba.py`       | WGAN convolucional        |
| `tfg_wgangp_celeba.py`          | WGAN-GP                   |

#### CycleGAN

`tfg_cyclegan.py` implementa la versión de Zhu et al. (2017): generador
ResNet de 9 bloques, discriminador PatchGAN de 70×70, pérdida adversaria
más pérdida cíclica (λ=10) más pérdida de identidad (λ=5). Optimizador
Adam con decay lineal del learning rate a partir de la época 100 (de las
200 totales). Se entrena sobre apple2orange.

#### Benchmark de tiempos

`benchmark_tiempos.py` mide cuánto tarda una época de entrenamiento para
cada combinación modelo por dataset. Hace 1 época de warmup que no se
cronometra y 5 épocas medidas, con `torch.cuda.synchronize()` antes y
después de cada una. No calcula FID/IS ni guarda imágenes para que la
medida refleje únicamente el coste del entrenamiento adversario.

`analyze_benchmark.py` que se encuentra en /ResultadosCódigosGeneralización/Tiempos
toma los resultados anteriores y genera la tabla LaTeX y las gráficas de tiempo por época,
tiempo por batch y escalado con la resolución.

## Métricas registradas

Cada entrenamiento genera un `metrics_all_runs.csv` con las columnas:

| Columna     | Descripción                                                 |
|-------------|-------------------------------------------------------------|
| `epoch`     | Número de época                                             |
| `run`       | Identificador de la ejecución (1…N)                         |
| `loss_g`    | Pérdida del generador                                       |
| `loss_d`    | Pérdida del discriminador (modelos con BCE)                 |
| `loss_c`    | Pérdida del crítico (modelos Wasserstein)                   |
| `gp`        | Término de gradient penalty (solo WGAN-GP)                  |
| `fid`       | FID, calculado cada `CALC_METRICS_FREQ` épocas              |
| `is_mean`   | Inception Score, media                                      |
| `is_std`    | Inception Score, desviación estándar                        |

Para CycleGAN no se guarda IS pues no se calcula y se guardan también:
`loss_GAN`, `loss_cycle`, `loss_identity`, `loss_D_A`, `loss_D_B`, `fid_AB`, `fid_BA` y `lr`


## Requisitos

- Python ≥ 3.9
- PyTorch ≥ 2.0 con soporte CUDA
- torchvision, torchmetrics
- pandas, numpy, matplotlib, Pillow

```bash
pip install torch torchvision torchmetrics pandas numpy matplotlib pillow
```

Para CelebA, descargar el dataset y dejarlo en un directorio con
estructura compatible con `ImageFolder` (las imágenes dentro de una
subcarpeta cualquiera; la ruta concreta se ajusta en `DATA_DIR` dentro
de cada script). Para apple2orange (CycleGAN), la estructura esperada es
`trainA/`, `trainB/`, `testA/`, `testB/`.


## Uso

Cada script se ejecuta de forma independiente:

```bash
cd CodigosRefactorizados
python tfg_dcgan_general.py          # DCGAN sobre el dataset que indique DATASET_NAME
python tfg_wgan_conv_general.py      # WGAN-Conv sobre el dataset que indique DATASET_NAME
python tfg_wgangp_celeba.py          # WGAN-GP sobre CelebA
python tfg_cyclegan.py               # CycleGAN sobre apple2orange
python benchmark_tiempos.py          # Tiempos de entrenamiento
python analyze_benchmark.py          # Tablas y gráficas del benchmark
```

Los resultados se vuelcan en el directorio indicado por `EXPERIMENT_DIR`
en cada script, con la estructura:

```
experimento/
├── images/    Cuadrículas de imágenes generadas durante el entrenamiento.
├── models/    Pesos finales (G y D/C) del primer run.
├── logs/      CSVs con las métricas por época y promediadas.
└── plots/     Curvas de pérdida, FID e IS (individuales y promediadas).
```

## Apunte sobre códigos de plot
Las imágenes introducidas en el TFG son tomadas de esos scripts, pero,
algunas fueron editadas después manualmente pues resultaba más rápido cambiar,
pequeños detalles manualmente que a código.

## Referencias

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
