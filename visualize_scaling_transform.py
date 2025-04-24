from music2latent.hparams import hparams
from music2latent.audio import normalize_complex, wv2complex
import torchaudio
import torch
alpha_rescale = 0.65
beta_rescale = 0.34

hparams.update({
    'alpha_rescale': alpha_rescale,
    'beta_rescale': beta_rescale,
    'data_paths': [],
    'data_path_test': []
})
wv, sr = torchaudio.load('./test2.wav')

stft_orig = wv2complex(wv).squeeze(0)
stft_norm = normalize_complex(stft_orig)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow(torch.log1p(stft_orig.abs()).numpy(), aspect='auto', origin='lower')
ax[0].set_title("Original STFT Magnitude")
ax[1].imshow(torch.log1p(stft_norm.abs()).numpy(), aspect='auto', origin='lower')
ax[1].set_title("Rescaled STFT Magnitude")
plt.tight_layout()
plt.show()