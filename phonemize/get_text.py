import IPython.display as ipd
import torch

import F0_MB_iSTFT_VITS.commons as commons
import F0_MB_iSTFT_VITS.utils as utils
from F0_MB_iSTFT_VITS.models import SynthesizerTrn
from F0_MB_iSTFT_VITS.text.symbols import symbols
from F0_MB_iSTFT_VITS.text import text_to_sequence

from phonemize import Tokenizer, auto_g2p


class Get_text:
    def __init__(self):
        self.tok = Tokenizer()
        self.auto_g2p = auto_g2p( )

    def get_text(self, text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)

        ipa_tokens, lang = self.auto_g2p(text)
        x, x_tst_lengths = self.tok(ipa_tokens)
        return x, x_tst_lengths, lang

