# Injury Atlas\n\nA repository for injury analysis and visualization.


# Gene Perturbation Prediction Pipeline

I am developing a pipeline to analyze a mouse single-cell RNA-seq atlas containing various cell types under injury and non-injury conditions, using condition labels for injury (injured vs non-injured) and cell state labels for perturbation (perturbed vs non-perturbed). The dataset includes multiple conditions, timepoints, and has been pre-processed with PCA embeddings batch effect corrected using Harmony. The metadata contains information about conditions, publication of origin, sequencing run, and cell type annotations. The goal is to identify key genes that can alter cell phenotypes from condition labels through overexpression and knockout perturbations. The workflow begins with data processing using Scanpy for quality control, normalization, and exploratory analysis. The dataset is split into training (50%), validation (25%), and holdout (25%) sets to ensure robust model evaluation. Then, I'll apply pre-trained models (GeneFormer) and gene perturbation models (scGen, scLAMBDA, scPRAM) to predict how specific genetic perturbations might alter condition labels across different tissues. The models will be fine-tuned on the training set and evaluated on both validation and holdout sets to ensure generalizability. The core approach involves using a cell's initial expression profile \(x\) and perturbation target \(p\) to predict the post-perturbation state \(x' = model(x, p)\), where \(p\) can represent either overexpression (increasing gene expression) or knockout (reducing gene expression to zero). To identify important genes, I'll implement a Multi-target Random Forest model to compare predicted expression profiles with initial states, enabling the identification of key regulatory genes and potential therapeutic targets. This is a multilabel problem because we're predicting multiple gene expression values simultaneously, where each gene's expression is a separate target in the prediction task. The pipeline will specifically focus on identifying genes whose overexpression or knockout can drive cells from injured to non-injured states or vice versa, with model performance validated across all three dataset splits.



## Pathway Analysis
- Use molecular data (e.g., RNA-seq from perturbed cells) to understand affected signaling pathways
- Compare with GRN/PPI network

## Reference
- [Overview Paper](https://www.biorxiv.org/content/10.1101/2024.12.23.630036v1.full)

## Available Tools and Frameworks

### Perturbation Screen Data
- [GenePert](https://github.com/zou-group/GenePert)

### Trainable Models
- **[scGen](https://github.com/theislab/scgen)** #Using
  - [Tutorial](https://scgen.readthedocs.io/en/stable/tutorials/scgen_perturbation_prediction.html)
- **[scLAMBDA](https://github.com/gefeiwang/scLAMBDA)** #Using
- [scELMO](https://github.com/HelloWorldLTY/scELMo) #Needs OpenAI key
- [trVAE](https://github.com/theislab/trVAE) #Dead
- **[scPARM](https://github.com/jiang-q19/scPRAM)**  #Using


### Pre-trained Models
- [Evo-131k](https://huggingface.co/togethercomputer/evo-1-131k-base) #Can't use
- [Evo-8k](https://huggingface.co/togethercomputer/evo-1-8k-base) #Can't use
- [scBERT](https://huggingface.co/tdc/scBERT) #Can't use 
- [Isoformer](https://huggingface.co/isoformer-anonymous/Isoformer) #Can't use Isoformer 
- [scGPT](https://github.com/bowang-lab/scGPT) #Needs OpenAI key
- [scPerb](https://github.com/QSong-github/scPerb) #Don't like this one
- [scFoundation](https://github.com/biomap-research/scFoundation) #Taken down
- **[GeneFormer](https://huggingface.co/ctheodoris/GeneFormer)** #Using

### Gene Regulatory Networks
- [GEARS](https://github.com/dmis-lab/GPO-VAE)

## Additional Resources
- [Twitter Discussion](https://x.com/mariosgeorgakis/status/1914713794091123127?s=42)
- [Lab Notes: Comparing GenePT and scGPT](https://learning-exhaust.hashnode.dev/lab-notes-comparing-genept-and-scgpt)


[Scanpy stuff](https://docs.scvi-tools.org/en/stable/tutorials/index_scrna.html)
-mrVI
