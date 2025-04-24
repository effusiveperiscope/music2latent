
import torchaudio
from music2latent.audio import wv2realimag, realimag2wv
import soundfile as sf

wv, sr = torchaudio.load('./test.wav')
X = wv2realimag(wv)
wv2 = realimag2wv(X)

#import pdb ; pdb.set_trace()
sf.write('./test_rec.wav', wv2.squeeze().cpu().numpy(), sr)