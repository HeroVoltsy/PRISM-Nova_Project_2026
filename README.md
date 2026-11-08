# PRISM
### Predictive Robustness and Invariant Structure in Multisite Neuroimaging

**A computational study of predictive performance, cross-site generalization, and robustness in autism spectrum disorder classification using resting-state functional connectivity.**

---

## Overview

Machine-learning models can identify predictive patterns in functional connectivity data. However, strong classification performance on familiar participants does not necessarily indicate that a model will generalize to data acquired at an unseen neuroimaging site.

**PRISM** investigates this distinction using the Autism Brain Imaging Data Exchange I (ABIDE-I) dataset. It evaluates machine-learning classifiers across participant-level evaluation and Leave-One-Site-Out (LOSO) validation, then examines their sensitivity to controlled input perturbations.

Rather than treating classification accuracy as the sole measure of model performance, this study considers three related but distinct dimensions:

1. **Predictive performance:** How effectively can models distinguish ASD participants from typically developing controls?
2. **Cross-site generalization:** How well do models perform when evaluated on participants from an acquisition site excluded entirely from training?
3. **Input robustness:** How does model performance and prediction behavior change when connectivity features are subjected to controlled perturbations?

The central motivation is to investigate whether predictive performance remains reliable under changes in acquisition environment and input representation.

## Research Objectives

PRISM aims to:

- Evaluate multiple machine-learning classifiers using resting-state functional connectivity features.
- Compare participant-level evaluation with site-held-out validation.
- Quantify variation in model performance across acquisition sites.
- Investigate the sensitivity of classifiers to Gaussian noise, feature dropout, random feature corruption, and distribution shift.
- Examine paired model comparisons and statistical uncertainty.
- Establish a reproducible computational framework for evaluating machine-learning reliability in multisite neuroimaging.

## Dataset

This study uses the **ABIDE-I** dataset.

| Characteristic | Description |
|---|---:|
| Dataset | ABIDE-I |
| Functional connectivity atlas | CC200 |
| Acquisition sites | 20 |
| Included participants | 989 |
| ASD participants | 480 |
| Typically developing controls | 509 |
| Connectivity features | 19,900 |
| Feature representation | Fisher-z-transformed Pearson correlations |

The connectivity features represent the unique pairwise relationships between 200 regions of interest (ROIs), extracted from resting-state functional MRI data.

The analytical cohort was established through participant identification, connectivity extraction, quality-control checks, and phenotype reconciliation.

### Data access

ABIDE-I data are available through the [ABIDE Preprocessed Initiative](http://preprocessed-connectomes-project.org/abide/).

Users must obtain the dataset through the appropriate data-access procedures. This repository does not redistribute the original participant-level neuroimaging data.

## Methodology

### 1. Data processing

The preprocessing and cohort-construction workflow includes:

- CC200 ROI time-series extraction and validation.
- Functional connectivity computation.
- Fisher z-transformation of correlation coefficients.
- Participant and phenotype reconciliation.
- Connectivity quality-control checks.
- Construction of the final analytical cohort.

### 2. Exploratory data analysis

Exploratory analyses examine:

- Participant and diagnostic-group distributions.
- Acquisition-site representation.
- Time-series characteristics.
- Connectivity feature distributions.
- Participant-level quality-control measurements.
- Potential site-related differences in diagnostic composition.

### 3. Machine-learning evaluation

The evaluated classifiers include:

- Logistic Regression
- Ridge Classifier
- Lasso Classifier
- Support Vector Classifier
- Extra Trees
- LightGBM
- XGBoost
- CatBoost
- HistGradientBoosting

The archived participant-level evaluation contains nine models. The principal LOSO analysis and current stress-testing experiment cover four models: Logistic Regression, Ridge Classifier, Lasso Classifier, and Support Vector Classifier.

### 4. Validation strategies

Two evaluation settings are considered.

**Participant-level evaluation**

Participants are evaluated using the recorded participant-level data splits. This provides a baseline assessment of classification performance under the implemented splitting procedure.

**Leave-One-Site-Out validation**

Each acquisition site is held out in turn. Models are trained using the remaining sites and evaluated exclusively on participants from the held-out site.

This approach provides an assessment of generalization to acquisition environments excluded from model training.

### 5. Statistical analysis

The statistical analysis includes descriptive performance summaries, paired model comparisons, and uncertainty estimation.

The implemented analyses include:

- McNemar tests for paired classification errors.
- DeLong tests for ROC-AUC comparisons.
- Holm multiple-comparison correction.
- Site-cluster bootstrap analysis.
- Quantile regression.
- Effect-size and uncertainty summaries.

Statistical results are interpreted alongside the validation design and the dependence structure of multisite observations.

### 6. Synthetic stress testing

Four controlled perturbation mechanisms are implemented in the original-feature LOSO stress-testing experiment.

| Perturbation | Tested severity |
|---|---|
| Gaussian noise | 0.01, 0.05, 0.10, 0.20 |
| Feature dropout | 5%, 10%, 20%, 40% |
| Random feature corruption | 5%, 10%, 20%, 40% |
| Distribution shift | 0.25, 0.50, 1.00, 2.00 SD |

Five perturbation replicates are used for each condition.

The perturbations are applied to held-out inputs while model preprocessing and classifier fitting are based on the original training data. The experiment examines performance changes, prediction flips, and site-specific sensitivity.

These perturbations are synthetic experimental conditions. They are not assumed to reproduce actual scanner failures or acquisition artifacts.

## Preliminary Findings

The archived experiments provide the following descriptive observations:

- Lasso Classifier achieved the highest recorded mean accuracy and balanced accuracy in the participant-level evaluation, both at 0.6902.
- Lasso Classifier also achieved the highest mean site-level balanced accuracy in the reconstructed LOSO analysis, at 0.6757.
- LOSO performance varied across acquisition sites, demonstrating the importance of examining site-level performance rather than relying exclusively on aggregate metrics.
- Paired statistical comparisons did not establish statistically significant superiority of Lasso Classifier over the other principal LOSO models after multiple-comparison correction.
- Synthetic perturbation effects varied across classifiers and perturbation mechanisms.

These findings are descriptive and should not be interpreted as evidence of clinical diagnostic validity or definitive biological differences between diagnostic groups.

**Reproducibility note:** The original LOSO predictions and the separate stress-testing baseline predictions do not fully agree. Stress-test degradation is therefore calculated relative to the stress-testing experiment's own baseline, not directly against the original LOSO evaluation. The discrepancy remains an implementation-level issue for further investigation.

## Repository Structure

The principal project directories are organized as follows:

```text
PRISM/
│
├── data/
│   └── ABIDE_I/
│       └── connectivity/
│           ├── X_connectivity.npy
│           └── metadata.csv
│
├── scripts/
│   ├── 00_CC200_ROI.py
│   ├── 01_validate_cc200.py
│   ├── 02_extract_connectivity.py
│   ├── 03_failure_audit.py
│   ├── 04_reconcile_cohort.py
│   ├── 05_failure_manifest.py
│   └── 07_save_cohort.py
│
├── notebooks/
│   └── 08_data_analysis_visualization.ipynb
│
├── outputs/
│   ├── model_experiments/
│   │   ├── ablation_results/
│   │   ├── loso_results/
│   │   ├── statistical_analysis/
│   │   └── model_comparison.csv
│   │
│   └── stress_testing/
│       └── loso_original/
│           ├── stress_test_audit.json
│           ├── loso_stress_predictions.csv
│           ├── loso_stress_metrics_by_repeat.csv
│           ├── loso_stress_site_summary.csv
│           └── loso_stress_overall_summary.csv
│
├── requirements.txt
├── references.bib
└── README.md
```

*This is a representative structure of the current project artifacts; individual files and directories may change as the implementation develops.*

## Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd PRISM
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The project uses Python-based scientific computing and machine-learning libraries, including NumPy, Pandas, SciPy, scikit-learn, and Matplotlib. Additional model-specific dependencies may be required depending on the experiments being executed.

## Data Preparation

Place the downloaded and appropriately processed ABIDE-I connectivity data in the expected project directory:

```text
data/ABIDE_I/connectivity/
```

The principal inputs are:

- `X_connectivity.npy`: participant-by-feature connectivity matrix.
- `metadata.csv`: participant-level diagnostic and acquisition-site metadata.

The matrix and metadata must have matching participant ordering. Confirm this before running any experiment.

## Running the Experiments

The processing scripts should be executed in their intended order, beginning with ROI extraction and validation, followed by connectivity extraction and cohort reconciliation.

The stress-testing implementation is available as:

```text
12_stress_testing_loso.py
```

Its expected inputs are:

```text
data/ABIDE_I/connectivity/X_connectivity.npy
data/ABIDE_I/connectivity/metadata.csv
```

Run it from the project root after verifying the configured paths and dependencies:

```bash
python 12_stress_testing_loso.py
```

The experiment writes its outputs to:

```text
outputs/stress_testing/loso_original/
```

Other model-evaluation and statistical-analysis workflows should be executed using their corresponding experiment notebooks and scripts. Consult the individual files for their exact configuration and execution requirements.

## Evaluation Metrics

The study considers the following metrics where applicable:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Balanced accuracy | Average recall across diagnostic classes |
| Precision | Proportion of positive predictions that are correct |
| Recall / Sensitivity | Proportion of ASD cases correctly identified |
| Specificity | Proportion of controls correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Matthews correlation coefficient | Balanced measure of binary classification quality |
| ROC-AUC | Discrimination across classification thresholds |
| PR-AUC | Precision-recall performance |

Stress-testing analyses additionally examine changes in performance relative to baseline and participant-level prediction-flip rates.

## Reproducibility and Limitations

Several considerations are important when interpreting or reproducing the experiments:

1. **Dataset scope:** The analysis uses ABIDE-I and does not independently validate performance on ABIDE-II or other external cohorts.
2. **Representation:** The principal feature representation consists of CC200 resting-state functional connectivity measurements.
3. **Validation:** Participant-level evaluation and LOSO address different generalization questions and should not be treated as interchangeable.
4. **Synthetic perturbations:** The stress-testing conditions do not constitute validated simulations of real scanner or acquisition failures.
5. **Statistical inference:** Comparisons involving multiple acquisition sites must account for site-level dependence and the limited number of sites.
6. **Implementation consistency:** Model configurations, preprocessing, and prediction outputs should be audited before results from separate experiment runs are combined.
7. **Clinical interpretation:** The models are research classifiers, not validated diagnostic instruments.

All reported results should be interpreted within these methodological limitations.

## Research Scope

PRISM is a computational machine-learning study. Its primary contribution is the evaluation of predictive performance, generalization, and robustness in multisite functional connectivity classification.

It does not attempt to establish new neurological mechanisms, infer causal biological relationships, or provide clinical diagnoses. Biological interpretation is limited to findings supported by published literature and the documentation of the underlying dataset.

## References

Relevant scientific and methodological literature includes:

- Di Martino et al. (2014). The Autism Brain Imaging Data Exchange.
- Nielsen et al. (2013). Multisite functional connectivity MRI classification of autism.
- Abraham et al. (2017). Deriving reproducible biomarkers from multi-site resting-state data.
- Craddock et al. (2012). A whole brain fMRI atlas generated via spatially constrained spectral clustering.
- Heinsfeld et al. (2018). Identification of autism spectrum disorder using deep learning and the ABIDE dataset.
- Varma and Simon (2006). Bias in error estimation when using cross-validation for model selection.
- Hendrycks and Dietterich (2019). Benchmarking neural network robustness to common corruptions and perturbations.
- Ovadia et al. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift.

Additional bibliographic information is maintained in `references.bib`.

## License and Data Use

A repository license should be added before public distribution. The original ABIDE-I data remain subject to their respective access and usage terms. Users must comply with the dataset's applicable data-use requirements and privacy protections.

---

**PRISM — Predictive Robustness and Invariant Structure in Multisite Neuroimaging**

*Evaluating not only whether a model can predict, but whether its predictions remain reliable under changing data conditions.*
