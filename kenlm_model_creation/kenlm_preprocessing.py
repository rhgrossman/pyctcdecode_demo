import pandas as pd
import nltk
from tqdm import tqdm

SPGISPEECH_TRAIN_CSV = "/data-ssd-2/speech_data/spgispeech/train.csv"
HUGGINGFACE_DEMO_VOCAB_SET =\
['_', '<s>', '</s>', '<unk>', ' ', 'E', 'T', 'A', 'O', 'N', 'I',
 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B',
 'V', 'K', "'", 'X', 'J', 'Q', 'Z']

# I just find lowercase easier to read
HUGGINGFACE_DEMO_VOCAB_SET = \
    [c.lower() for c in HUGGINGFACE_DEMO_VOCAB_SET]

NEMO_DEMO_VOCAB_SET = \
    [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
     'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
     'z', "'"]

# These two parameters must be set to match your model
# The two vocab sets above match the models used in demo-huggingface
# and demo-nemo
CURRENT_VOCAB_SET = HUGGINGFACE_DEMO_VOCAB_SET #NEMO_DEMO_VOCAB_SET
lowercase_text = True
uppercase_text = False

df = pd.read_csv(SPGISPEECH_TRAIN_CSV, sep="|")

for inx, row in tqdm(df.iterrows()):
    line = row['transcript']
    for sentence in nltk.sent_tokenize(line):

        # Make our text lowercase if our model is trained lowercase
        if lowercase_text:
            sentence = sentence.lower()

        # hyphens will mess up words since they are not postpended
        # by a space
        if "-" not in CURRENT_VOCAB_SET:
            sentence = sentence.replace("-", " ")


        # Remove any extraneous characters not in our vocab set
        vocab_matched_sentence = \
            ''.join(c for c in sentence if c in CURRENT_VOCAB_SET)

        print(' '.join(nltk.word_tokenize(sentence)))

