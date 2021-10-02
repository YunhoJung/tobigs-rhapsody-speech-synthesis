import librosa


def pitch_up(y, sr = 44100, n_step = 10):
    y_pitch_higher = librosa.effects.pitch_shift(y, sr, n_steps=n_step)
    return y_pitch_higher


def pitch_down(y, sr = 44100, n_step = -10):
    y_pitch_lower = librosa.effects.pitch_shift(y, sr, n_steps=n_step)
    return y_pitch_lower


def speed_up(y, n_step = 2):
    y_stft = librosa.stft(y)
    y_stft_fast = librosa.phase_vocoder(y_stft, n_step)
    y_faster = librosa.istft(y_stft_fast)
    return y_faster


def speed_down(y, n_step=0.5):
    y_stft = librosa.stft(y)
    y_stft_slow = librosa.phase_vocoder(y_stft, n_step)
    y_slower = librosa.istft(y_stft_slow)
    return y_slower
