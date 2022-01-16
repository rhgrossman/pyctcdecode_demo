# For this demo, it is assumed that you have access to a kenlm
# language model trained on a relevant corpus to the one you are
# predicting on.
# You can create a language model using SPGISpeech by following the
# instructions in the readme in kenlm_model_creation

import kenlm
import pandas as pd
from pyctcdecode import build_ctcdecoder
from pydub import AudioSegment
from pydub.playback import play
import random
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from utils import greedy_decode

KENLM_MODEL_LOC = "/home/raymond/demos/data/demo_huggingface_spgispeech.arpa"
SPGI_VAL_DIR = "/data-ssd-2/speech_data/spgispeech/val/"
SPGI_VAL_CSV = "/data-ssd-2/speech_data/spgispeech/val.csv"

# Load the val csv
val_df = pd.read_csv(SPGI_VAL_CSV, sep='|')

asr_processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h")
print("Vocab: ", asr_processor.tokenizer.get_vocab())

# Make vocab more human readable
# Replace <pad> character with placeholder '_'
# Replace '|' with ' '
# This is done for compatability with the greedy decode function
# which is based off characters TODO @ray rewrite gd
vocab = list(asr_processor.tokenizer.get_vocab().keys())
vocab[0] = '_'
vocab[4] = ' '
# Because I find lowercase easier to read
vocab = \
    [c.lower() for c in vocab]


decoder = build_ctcdecoder(
    labels = vocab,
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

    # play
    arr, _ = sf.read(sample_loc)

    input_values = asr_processor(arr, return_tensors="pt", sampling_rate=16000).input_values  # Batch size 1
    logits = asr_model(input_values).logits.cpu().detach().numpy()[0]

    # get greedy decoding

    greedy_text = greedy_decode(logits, vocab)
    greedy_text = ("".join(c for c in greedy_text if c not in ["_"]))
    text = decoder.decode(logits)

    print("Sample: ", sample_name)
    print("\n")
    print("Greedy Decoding: \n" + greedy_text)
    print("\n")
    print("Language Model Decoding: \n" + text)
    print("\n")
    print("Ground truth \n" + true_text)
    print("\n")