import IPython.display as ipd
import torch
import F0_MB_iSTFT_VITS.utils as utils
from F0_MB_iSTFT_VITS.models import SynthesizerTrn
from F0_MB_iSTFT_VITS.text.symbols import symbols
from phonemize import Get_text

text = "This is a sample audio"
model_path = "F0_MB_iSTFT_VITS/logs/ljs_mb_istft_vits/G_800000.pth"
hps = utils.get_hparams_from_file("F0_MB_iSTFT_VITS/configs/ljs_mb_istft_vits.json")

class kasane_tts:
    def __init__(self):
        self.net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model
        ).cuda().eval()

        _ = utils.load_checkpoint(model_path, self.net_g, None)
        self.get = Get_text()

    def __call__(self, text, f0_curve=None):
        with torch.no_grad():
            x_tst, x_tst_lengths, lang = self.get(text, hps)

            audio = self.net_g.infer(
                x_tst,
                x_tst_lengths,
                lang_id=lang,
                noise_scale=.667,
                noise_scale_w=0.8,
                length_scale=1,
                f0_external=f0_curve
            )[0][0,0].data.cpu().float().numpy()
 
        ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))


if __name__ == "__main__":
    kasane = kasane_tts()
    kasane("天気が良いから散歩しよう。")
