# HCMR

Code and data for the paper "Diagnosing and repairing encoding collapse in resource-limited quantum neural networks", submitted to npj Quantum Information.

## Setup

```
pip install -r requirements.txt
```

MNIST, Fashion-MNIST and CIFAR-10 are downloaded to `data/` on first use. The ibm_fez run needs an IBM Quantum account saved with `qiskit-ibm-runtime`.

## Data

`paper_data/` holds the results reported in the paper, one row per run (5 runs per configuration).

| file | content |
|---|---|
| `diagnosis_per_sample.csv` | per test sample: label, HEM, CE flag, predictions of the reference QNN (Z_k and Z+ZZ readout), the parameter-matched MLP, and the raw-pixel and embedded-state probes |
| `diagnosis_probes.csv` | accuracy of the classical probes on each pipeline stage |
| `diagnosis_mlp_runs.csv` | parameter-matched MLP on the embedded state |
| `main_runs.csv` | all encoders on the 48 settings: accuracy, CE / PE accuracy, CE share (`ce_size`), selected margins and joint-training length |
| `margin_grid.csv` | validation CE share (`val_ce_size`) of every margin candidate in the encoder fitting, with the selected one marked |
| `j_sweep.csv` | HCMR and HCMR without margin pretraining for a dense grid of joint-training lengths |
| `joint_window.csv` | HCMR with one joint-training window at different positions |
| `layers_runs.csv` | reference QNN with modified PQC depth and readout |
| `noise_training_runs.csv` | HCMR trained under shot noise, device noise, or both |
| `noise_gradient.csv` | cosine similarity and norm ratio between noisy and exact gradients per epoch |
| `noise_inference_runs.csv` | trained models evaluated with shot noise, device noise, or both (FakeFez, FakeNighthawk), with TVD and 2-qubit gate counts |
| `noise_model_calibration.csv` | calibration medians of FakeFez, FakeNighthawk and the ibm_fez runs; parameters of the training-noise model |
| `device_ibm_fez.csv` | ibm_fez runs and the noisy simulation of the same circuits, with flipped samples and decision margins |
| `device_calibration/` | ibm_fez calibration at the time of each job (backend properties), one file per embedding and variant |

## Code

- `hcmr/`: data, embeddings and HEM, encoders, QNN simulator, noise simulation
- `scripts/`: experiments
- `analysis/`: figures, tables and `paper_data/`

Run order:

```
python scripts/train_main.py --stage fit --device cuda
python scripts/train_main.py --stage train --workers 12
python scripts/train_main.py --stage train --methods hcmr,hcmr_cnn,wo_pretrain,cnn_wo_pretrain --qubits 6 --j-grid 0,5,10,15,20,30,40,50,60 --out results/j_sweep --enc-dir results/full
python scripts/train_main.py --stage train --methods hcmr,hcmr_cnn --qubits 6 --window-starts 0,10,20,30,40,50 --out results/joint_window --enc-dir results/full
python scripts/train_layers.py --workers 12
python scripts/probes.py
python scripts/train_mlp.py --workers 12

python scripts/train_main.py --stage train --methods hcmr,hcmr_cnn --qubits 4,8 --j-grid 10 --save-models --out results/noise/models --enc-dir results/full
python scripts/noise_infer.py
python scripts/count_state_prep.py
python scripts/calibrate_prep.py
python scripts/noise_train.py --device cuda
python scripts/noise_grad.py --setups amplitude_mnist_8q4c,amplitude_fmnist_8q4c

python scripts/device_fez.py prepare
python scripts/device_fez.py submit
python scripts/device_fez.py fetch
python scripts/device_fez.py analyze
```

Then any script in `analysis/`. Without `--seeds`, seeds are drawn at random and stored in each result file; individual numbers vary with the seed and the hardware, the trends do not.

## License

MIT
