# Evaluating Gemma Models for Causal Discovery in Vehicle Telematics

**Bachelor's Thesis (Computer Science) — Constructor University, 2026**  
**Author:** Gabil Majidov  
**Tools:** Python, JupyterLab, Ollama, pandas, NetworkX, NumPy, scikit-learn, SciPy, matplotlib, seaborn

[Read the bachelor's thesis (Google Drive)](https://drive.google.com/file/d/10JmZpgEm_-PnQ2NTInCiC3-wGpPboCYO/view)

## Overview

This project investigates whether locally run Gemma language models can infer directed causal relationships among selected hybrid-vehicle telematics variables. The experiments compare **Gemma 2B** and **Gemma 7B** across **five prompting strategies**: zero-shot, role, emotion, style, and contextual prompting.

The models are prompted to return causal pairs such as `feature_a -> feature_b`. These relationships are parsed into directed graphs and adjacency matrices, then compared with a manually specified, domain-informed reference graph. The aim is to evaluate structural agreement and output reliability, **not** to predict vehicle energy consumption.

## Research Workflow

1. Prepare selected variables from normalized vehicle telematics data.
2. Construct a reference causal graph and adjacency matrix.
3. Run five prompting strategies separately on Gemma 2B and Gemma 7B through a local Ollama API.
4. Parse generated relationships into directed graphs and adjacency matrices.
5. Compare model outputs to the reference using precision, recall, F1-score, Structural Hamming Distance (SHD), False Discovery Rate (FDR), and matrix similarity metrics.
6. Summarize recorded experiment results.

The thesis found that **Gemma 7B with contextual prompting** performed best overall in the reported experiments, while **role prompting** was the strongest configuration within the Gemma 2B group. Performance and schema validity varied across runs.

## Repository Contents

| Path | Description |
|---|---|
| `notebooks/01_ground_truth.ipynb` | Constructs the domain-informed reference graph and adjacency matrix |
| `notebooks/02_gemma_2b.ipynb` | Five prompting strategies evaluated with Gemma 2B |
| `notebooks/03_gemma_7b.ipynb` | Five prompting strategies evaluated with Gemma 7B |
| `notebooks/04_adjacency_matrix_comparison.ipynb` | Calculates structural metrics for a selected saved adjacency matrix |
| `notebooks/05_final_results.ipynb` | Aggregates recorded, manually entered per-run metrics and plots summary results |
| `data/README.md` | Explains required input columns and why the original dataset is excluded |

## Software Requirements

The experiments were developed using **Python 3.14.3**, **JupyterLab 4.5.0**, and **Ollama 0.21.1**.

The main Python libraries used in the project are:

| Library | Purpose |
|---|---|
| `pandas` | Loading, preprocessing, and analyzing vehicle telematics data |
| `numpy` | Numerical operations and adjacency matrices |
| `networkx` | Constructing and analyzing directed causal graphs |
| `requests` | Communicating with the local Ollama API |
| `scikit-learn` | Evaluation metrics and data analysis |
| `scipy` | Statistical calculations and matrix comparisons |
| `matplotlib` | Creating charts and visualizations |
| `seaborn` | Statistical visualization and result heatmaps |
| `jupyterlab` | Interactive execution of Python notebooks |
| `ipykernel` | Python kernel support for Jupyter notebooks |

Install the Python dependencies using:

`python -m pip install jupyterlab ipykernel requests pandas numpy networkx matplotlib scipy scikit-learn seaborn`

**Ollama** must be installed separately to run the language models locally.

The required models are:

- `gemma:2b`
- `gemma:7b`

## Running Locally

1. Install Python, JupyterLab, and Ollama.
2. Install the Python libraries listed in the Software Requirements section.
3. Download the required Ollama models using `ollama pull gemma:2b` and `ollama pull gemma:7b`.
4. Place a legally shareable or privately held authorized normalized telematics CSV in `data/telematics.csv`, with the columns listed in `data/README.md`.
5. Run Jupyter from the `notebooks/` directory so the notebooks' relative paths to `../data/` and `../results/` resolve.
6. Run the ground-truth notebook first, followed by the Gemma model notebooks. The comparison notebook illustrates the evaluation of one selected saved adjacency matrix.

**Important:** These are cleaned versions of the original research notebooks, with stored execution outputs removed and the input CSV path adjusted for this repository. They have **not been freshly executed or validated end-to-end** in this public package. The Ollama request settings, model versions, generated outputs, and repeated-run workflow should be checked before attempting to reproduce the results.

## Data Availability and Reproducibility

The original study used a normalized private vehicle telematics dataset containing **15,727 observations and seven selected variables**.

The dataset is **not included** because permission to redistribute it publicly has not been established.

Results are described in the bachelor's thesis. The `final_results` notebook summarizes per-run metric values recorded during the original experiments rather than recomputing all runs from raw model responses.

This is an academic research-code portfolio, not a production AI service or a fully automated reproduction package.
