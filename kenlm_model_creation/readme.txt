###

To download the SPGISpeech data needed for this tutorial, visit:
https://datasets.kensho.com/datasets/spgispeech

For more generic instructions on how to train a kenlm language model for use
in speech-to-text with a corpus that is not SPGISpeech, visit
https://github.com/kmario23/KenLM-training

### Training Instructions

To train a kenlm language model using the SPGISpeech data:

1. Change the train_csv location in kenlm_preprocessing to the
   correct location for your machine. Change the vocabulary set that is being used to one
   appropriate for the model you are using.

2. Run $   python kenlm_preprocessing.py |\
  /path/to/kenlm/bin/lmplz -o 3 > /path/to/spgispeech.arpa

  You can change the order here to your desired order by changing the -o value.

  eg. python /home/raymond/github/pyctc-demo/kenlm_model_creation/kenlm_preprocessing.py |\
  /home/raymond/github/kenlm/bin/lmplz -o 3 > /home/raymond/demos/data/demo_nemo_spgispeech.arpa

3. You can change the .arpa to a .binary if you desire for some speedup
    /path/to/kenlm/bin/build_binary bible.arpa /path/to/spgispeech.binary

   eg. /home/raymond/github/kenlm/bin/build_binary /home/raymond/demos/data/demo_nemo_spgispeech.arpa \
   /home/raymond/demos/data/demo_nemo_spgispeech.binary

python /home/raymond/github/pyctc-demo/kenlm_model_creation/kenlm_preprocessing.py |\
  /home/raymond/github/kenlm/bin/lmplz -o 3 > /home/raymond/demos/data/demo_huggingface_spgispeech.arpa
