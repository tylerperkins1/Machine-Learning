import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display


def specDisplay(file):

    y, sr = librosa.load(file)
    plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.waveplot(y, sr=sr)
    plt.title('WaveForm')
    plt.savefig('logs/waveform.png', format='png', bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig('logs/spectrogram.png', format='png', bbox_inches='tight')

    plt.figure(figsize=(15, 4));
    plt.subplot(1, 3, 2);
    mel_10 = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=10)
    librosa.display.specshow(mel_10, sr=sr, hop_length=512, x_axis='linear');
    plt.ylabel('Mel filter');
    plt.colorbar();
    plt.title('Mel spectrogram');
    plt.tight_layout();
    plt.savefig('logs/melspectrogram.png', format='png', bbox_inches='tight')