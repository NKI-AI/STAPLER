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
3. Install the `STAPLER` package
   ```sh
    python -m pip install .
   ```
### Data and model checkpoints

The following data is available <a href="https://files.aiforoncology.nl/stapler/" target="_blank">here</a>:
* TCR and peptide datasets used to pre-train STAPLER.
* 5 train and validation folds containing labeled TCR-peptide pairs used to fine-tune STAPLER.
* 1 pre-trained model checkpoint.
* 5 fine-tuned model checkpoints (one for each fold).
* VDJDB+ETN test dataset used to test STAPLER.

### Requirements
STAPLER was pre-trained and fine-tuned using an a100 GPU on a SLURM cluster. 
Pre-trained and fine-tuned versions of smaller model sizes will be added in the future, which will fit on smaller GPUs.

<!-- USAGE EXAMPLES -->
## Usage

### Pre-training, fine-tuning and testing of STAPLER

Inside the `tools` directory you should change the following file:
* `.env.example`: Environment file with paths to data, model checkpoints and output paths. You should adapt the `.env.example` to your local file-system and then change the file-name to `.env`. For more information see <a href="https://github.com/theskumar/python-dotenv" target="_blank">python-dotenv</a>.

Here you can also find the following files to pre-train, fine-tune and/or test STAPLER on a <a href="https://slurm.schedmd.com" target="_blank">SLURM</a> cluster.  Also provide an argument to `--partition` to specify the partition to use. 
* `sbatch pretrain_STAPLER.sh`: Pre-train STAPLER.
* `sbatch train_STAPLER.sh`: Fine-tune STAPLER using 5-fold cross-validation.
* `sbatch test_STAPLER.sh`: Test on a test set using a fine-tuned model checkpoint. 


### Custom parameters
If you want to experiment with alternative parameters, 
you can do so in the `config` directory (implemented using <a href="https://hydra.cc/docs/intro/" target="_blank">Hydra</a>). 
The `config` directory contains the following main configuration files:
* `pretrain.yaml`: Configuration parameters file for pre-training.
* `train_5_fold.yaml`: Configuration parameters file for fine-tuning.
* `test.yaml`: Configuration parameters file for testing.


### TO-DO's

- [ ] Add support for other model sizes (pre-trained and fine-tuned model checkpoints).

## Issues
To request a feature or to discuss any issues, please let us know by opening an issue on the <a href="https://github.com/NKI-AI/STAPLER/issues" target="_blank">issues page</a>.

<!-- CONTACT -->
## Contact

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




