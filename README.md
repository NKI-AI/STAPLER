<div id="top"></div>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/NKI-AI/STAPLER">
    <img src="STAPLER_logo.png" alt="Logo" height="200">
  </a>

<h3 align="center">STAPLER</h3>
    

  <p align="center">
    <a href="https://www.biorxiv.org/content/10.1101/2023.04.25.538237v1" onclick="window.open('#','_blank');window.open(this.href,'_self');"> Preprint (bioRxiv) </a>
    .
    <a href="https://github.com/NKI-AI/STAPLER/issues" onclick="window.open('#','_blank');window.open(this.href,'_self');">Report Bug</a>
  </p>
</div>



<!-- GETTING STARTED -->
## Getting Started

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/NKI-AI/STAPLER.git
   ```
2. Navigate to the directory containing `setup.py` 
    ```sh
    cd STAPLER
    ```
3. Install the `STAPLER` package (should take less than 10 minutes)
   ```sh
    python -m pip install .
   ```
   
### Data and model checkpoints

The following data is available <a href="https://files.aiforoncology.nl/stapler/" target="_blank">here</a>:

* data/train:
    * TCR and peptide datasets used to pre-train STAPLER
    * The training dataset
    * 5 train and validation folds created from the training dataset used to fine-tune STAPLER
* data/test:
  * VDJDB+ and VDJDB+ETN test datasets used to test STAPLER
* model/pretrained_model:
    * 1 pre-trained model checkpoint
* model/finetuned_model:
  * 5 fine-tuned model checkpoints (one for each fold).
* predictions:
  * 5 predictions for each fold on the VDJDB+ETN test set
  * 1 ensembled prediction of all 5-fold predictions on the VDJDB+ETN test set


### Requirements
STAPLER was pre-trained and fine-tuned using an a100 GPU. At this moment no other GPU's have been tested.

### Setup

Inside the `tools` directory the following file should be changed:
* `.env.example`: Which is an environment file with paths to data, model checkpoints and output paths. It should be adapt the `.env.example` to your local file-system and then change the file-name to `.env`. For more information see <a href="https://github.com/theskumar/python-dotenv" target="_blank">python-dotenv</a>.



<!-- USAGE EXAMPLES -->
## Usage

### Pre-training, fine-tuning and testing of STAPLER

Inside the `tools` directory contains the following files to pre-train, fine-tune and/or test STAPLER on a <a href="https://slurm.schedmd.com" target="_blank">SLURM</a> cluster.  Also provide an argument to `--partition` to specify the partition to use. 
* `sbatch pretrain_STAPLER.sh`: Pre-train STAPLER.
* `sbatch train_STAPLER.sh`: Fine-tune STAPLER using 5-fold cross-validation.
* `sbatch test_STAPLER.sh`: Test on a test set using a fine-tuned model checkpoint. 

Alternatively run STAPLER directly on a machine with an appropriate GPU (see requirements).
* `python pretrain.py`: Pre-train STAPLER.
* `python train_5_fold.py`: Fine-tune STAPLER using 5-fold cross-validation.
* `python test.py`: Test on a test set using a fine-tuned model checkpoint.

### Required GPU time
The pre-training should take a day, fine-tuning should take a couple of hours per fold and testing/inference should take a couple of minutes for all 5-fold predictions.

### Custom parameters
To experiment with custom model parameters 
change the paramteres inside the `config` directory (implemented using <a href="https://hydra.cc/docs/intro/" target="_blank">Hydra</a>). 
The `config` directory contains the following main configuration files:
* `pretrain.yaml`: Configuration parameters file for pre-training.
* `train_5_fold.yaml`: Configuration parameters file for fine-tuning.
* `test.yaml`: Configuration parameters file for testing.


## Issues and feature requests
To request a feature or to discuss any issues, please let us know by opening an issue on the <a href="https://github.com/NKI-AI/STAPLER/issues" target="_blank">issues page</a>.

- [ ] The notebooks used to make the pre-print figures will be available soon

<!-- CONTACT -->
## Contact

Corresponding author: <a href="https://www.nki.nl/research/find-a-researcher/groupleaders/ton-schumacher/" target="_blank">Ton Schumacher</a>

Ton Schumacher group (NKI) - <a href="https://www.nki.nl/research/research-groups/ton-schumacher/" target="_blank">Group website</a> - [![Twitter][twitter-shield_lab]][twitter-url_lab]

Ai for Oncology group (NKI) - <a href="https://www.aiforoncology.nl" target="_blank">Group website</a> - [![Twitter][twitter-shield_ailab]][twitter-url_ailab]


<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

The development of the STAPLER model is the result of a collaboration between the Schumacher lab AIforOncology lab at the Netherlands Cancer Institute. The following people contributed to the development of the model:
* Bjørn Kwee (implementation, development, evaluation, refactoring)
* Marius Messemaker (supervision, development, evaluation)
* Eric Marcus (supervision, refactoring)
* Wouter Scheper (supervision)
* Jonas Teuwen (supervision)
* Ton Schumacher (supervision)

A part of the data was provided, and consequent results were interpreted by the following people from the <a href="https://wulab.dfci.harvard.edu" target="_blank">Wu lab</a>  (DFCI and Harvard Medical School): 
* Giacomo Oliveira
* Catherine Wu

STAPLER is built on top of the <a href="https://github.com/lucidrains/x-transformers" target="_blank">x-transformers package</a>


<!-- LICENSE -->
## License

Distributed under the Apache 2.0 License. 


<p align="right">(<a href="#top">back to top</a>)</p>

[twitter-shield_lab]: https://img.shields.io/twitter/follow/Schumacher_lab?
[twitter-url_lab]:  https://twitter.com/intent/follow?screen_name=Schumacher_lab
[twitter-shield_ailab]: https://img.shields.io/twitter/follow/AI4Oncology?
[twitter-url_ailab]:  https://twitter.com/intent/follow?screen_name=AI4Oncology




