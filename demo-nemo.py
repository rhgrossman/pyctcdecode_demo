# For this demo, it is assumed that you have access to a kenlm
# language model trained on a relevant corpus to the one you are
# predicting on.
# You can create a language model using SPGISpeech by following the
# instructions in the readme in kenlm_model_creation

import kenlm
import nemo.collections.asr as nemo_asr
import pandas as pd
from pyctcdecode import build_ctcdecoder
from pydub import AudioSegment
from pydub.playback import play
import random

from utils import greedy_decode

KENLM_MODEL_LOC = "/home/raymond/demos/data/demo_nemo_spgispeech.arpa"
SPGI_VAL_DIR = "/data-ssd-2/speech_data/spgispeech/val/"
SPGI_VAL_CSV = "/data-ssd-2/speech_data/spgispeech/val.csv"

# Load the val csv
val_df = pd.read_csv(SPGI_VAL_CSV, sep='|')

# Load the ngram model -- testing only
# kenlm_model = kenlm.Model(KENLM_MODEL_LOC)

asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name='QuartzNet15x5Base-En')

print("Vocab: {}".format(asr_model.decoder.vocabulary))
decoder = build_ctcdecoder(
    labels = list(asr_model.decoder.vocabulary),
    kenlm_model_path = KENLM_MODEL_LOC,
    alpha=0.6,  # tuned on a val set
    beta=2.0,  # tuned on a val set
)

continue_looping = 1

# Select random items in our val set to listen to and predict while the user desires
while continue_looping:
    input("Press Enter to select a random sample ... ")

    # select random sample
    sample_number = random.randint(0, len(val_df))
    sample_name = val_df.loc[sample_number, "wav_filename"]
    true_text = val_df.loc[sample_number, 'transcript']
    sample_loc = SPGI_VAL_DIR + sample_name

    # listen to sample
    input("Press Enter to listen to audio...")
    audio = AudioSegment.from_wav(sample_loc)
    play(audio)
    input("Press Enter to continue...")

    logits = asr_model.transcribe([sample_loc], logprobs=True)[0]

    # get greedy decoding

    greedy_text = greedy_decode(logits, asr_model.decoder.vocabulary)
    text = decoder.decode(logits)

    print("Sample: ", sample_name)
    print("\n")
    print("Greedy Decoding: \n" + greedy_text)
    print("\n")
    print("Language Model Decoding: \n" + text)
    print("\n")
    print("Ground truth \n" + true_text)
    print("\n")
