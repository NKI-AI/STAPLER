<div id="top"></div>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/NKI-AI/STAPLER">
    <img src="STAPLER_logo.png" alt="Logo" height="200">
  </a>

<h3 align="center">STAPLER</h3>
    

  <p align="center">
    <a href="https://www.biorxiv.org/content/10.1101/2023.04.25.538237v1" onclick="window.open('#','_blank');window.open(this.href,'_self');"> Preprint (bioArxiv) </a>
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


<!-- USAGE EXAMPLES -->
## Usage

### Pre-training, fine-tuning and testing of STAPLER

Inside the `tools` directory you should change the following file:
* `.env`: Environment file with paths to data, output paths and model checkpoints. You should adapt the `.env.example` to your local file-system and then change the file-name to `.env`.

Here you can also find the following files to pre-train, fine-tune and/or test STAPLER on a SLURM cluster (currently only supports the STAPLER (medium sized) model). Also provide an argument to `--partition` to specify the partition to use.
* `sbatch pretrain_STAPLER.sh`: Pre-train STAPLER.
* `sbatch train_STAPLER.sh`: Fine-tune STAPLER using 5-fold cross-validation.
* `sbatch test_STAPLER.sh`: Test on a test set using a fine-tuned model checkpoint. 


### Custom parameters
If you want to experiment with alternative parameters, you can do so in the `config` directory. The `config` directory contains the following files:
* `pretrain.yaml`: Configuration parameters file for pre-training.
* `train_5_fold.yaml`: Configuration parameters file for fine-tuning.
* `test.yaml`: Configuration parameters file for testing.

## Issues
If you encounter any issues, please let us know by opening an issue on the <a href="https://github.com/NKI-AI/STAPLER/issues" target="_blank">issues page</a>.

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
* Jonas Teuwen (supervision)
* Ton Schumacher (supervision)

A part of the data was provided- and consequent results were interpreted by the following people from the <a href="https://wulab.dfci.harvard.edu" target="_blank">Wu lab</a>  (DFCI and Harvard Medical School): 
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




