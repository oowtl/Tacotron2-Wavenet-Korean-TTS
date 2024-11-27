# coding: utf-8

"""
1.	오디오 파일 처리
•	오디오 파일을 로드하고 저장하는 기능 제공 (load_wav, save_wav 등).
2.	오디오 신호 변환
•	프리엠퍼시스 적용 및 역변환을 통해 고주파 성분을 강조하거나 복원.
3.	스펙트로그램 처리
•	오디오 데이터를 선형 및 멜 스펙트로그램으로 변환하거나 이를 다시 오디오로 복원.
4.	무음 제거
•	오디오 신호의 앞뒤 무음 구간을 제거하여 데이터 정제.
5.	신호 압축 및 복원
•	Mu-law 압축 및 복원, 양자화 작업 제공.
6.	Griffin-Lim 알고리즘
•	위상 복원을 통해 스펙트로그램에서 오디오 데이터를 재구성.
7.	정규화 및 비정규화
•	스펙트로그램 데이터를 모델 학습에 적합한 범위로 정규화하거나 복원.
8.	TensorFlow 통합
•	TensorFlow를 사용해 Griffin-Lim 및 STFT(단시간 푸리에 변환)와 같은 신호 처리 수행.

전체적인 목적
오디오 데이터를 딥러닝 모델에서 활용하기 위해 전처리, 변환, 복원 작업을 체계적으로 처리하며, 음성 합성 및 변환 프로젝트(예: Tacotron, Wavenet)에 적합한 기능들을 제공합니다.
"""

# 필요한 라이브러리 임포트
import librosa  # 오디오 처리 라이브러리
import librosa.filters  # 필터 관련 함수
import numpy as np  # 수학적 연산을 위한 라이브러리
import tensorflow as tf  # 머신러닝 라이브러리
from scipy import signal  # 신호 처리 관련 라이브러리
from scipy.io import wavfile  # WAV 파일 입출력 라이브러리
# from tensorflow.contrib.training.python.training.hparam import HParams  # 하이퍼파라미터 관리 (TensorFlow 1.x)

# WAV 파일을 로드하여 지정된 샘플레이트로 반환
def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

# WAV 파일을 저장
def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))  # 음파를 16비트 정수 범위로 정규화
    wavfile.write(path, sr, wav.astype(np.int16))

# Wavenet용 WAV 파일을 저장
def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)

# Pre-emphasis 필터를 적용 (고주파 강조)
def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)  # 필터 적용
    return wav

# Pre-emphasis 필터를 제거
def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)  # 필터 제거
    return wav

# 음소거 구간의 시작과 끝 인덱스를 찾기
def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:  # 음소거 임계값 이상인지 확인
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    # 시작과 끝이 음소거 임계값 이상인지 검증
    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end

# WAV 파일에서 음소거를 제거
def trim_silence(wav, hparams):
    '''
    앞뒤 음소거를 제거
    '''
    return librosa.effects.trim(
        wav, top_db=hparams['trim_top_db'],
        frame_length=hparams['trim_fft_size'],
        hop_length=hparams['trim_hop_size'])[0]

# STFT에서 프레임 이동 크기(hop size)를 계산
def get_hop_size(hparams):
    hop_size = hparams['hop_size']
    if hop_size is None:  # hop_size가 없으면 계산
        assert hparams['frame_shift_ms'] is not None
        hop_size = int(hparams['frame_shift_ms'] / 1000 * hparams['sample_rate'])
    return hop_size

# 선형 스펙트로그램 계산
def linearspectrogram(wav, hparams):
    D = _stft(preemphasis(wav, hparams['preemphasis'], hparams['preemphasize']), hparams)
    S = _amp_to_db(np.abs(D), hparams) - hparams['ref_level_db']

    if hparams['signal_normalization']:  # 신호를 정규화
        return _normalize(S, hparams)
    return S

# 멜 스펙트로그램 계산
def melspectrogram(wav, hparams):
    D = _stft(preemphasis(wav, hparams['preemphasis'], hparams['preemphasize']), hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams), hparams) - hparams['ref_level_db']

    if hparams['signal_normalization']:
        return _normalize(S, hparams)
    return S

# 선형 스펙트로그램을 음파로 변환
def inv_linear_spectrogram(linear_spectrogram, hparams):
    if hparams['signal_normalization']:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram

    S = _db_to_amp(D + hparams['ref_level_db'])  # 스펙트로그램을 선형 값으로 변환

    if hparams['use_lws']:  # LWS 라이브러리를 사용하는 경우
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams['power'])
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams['preemphasis'], hparams['preemphasize'])
    else:  # Griffin-Lim 알고리즘을 사용하는 경우
        return inv_preemphasis(_griffin_lim(S ** hparams['power'], hparams), hparams['preemphasis'], hparams['preemphasize'])

# 멜 스펙트로그램을 음파로 변환
def inv_mel_spectrogram(mel_spectrogram, hparams):
    if hparams['signal_normalization']:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram

    S = _mel_to_linear(_db_to_amp(D + hparams['ref_level_db']), hparams)

    if hparams['use_lws']:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams['power'])
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams['preemphasis'], hparams['preemphasize'])
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams['power'], hparams), hparams['preemphasis'], hparams['preemphasize'])

# 텐서플로우를 사용한 스펙트로그램 역변환
def inv_spectrogram_tensorflow(spectrogram, hparams):
    S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram, hparams) + hparams['ref_level_db'])
    return _griffin_lim_tensorflow(tf.pow(S, hparams['power']), hparams)

def inv_spectrogram(spectrogram, hparams):
    # 스펙트로그램을 선형 값으로 변환 후 역변환하여 음파로 복원
    S = _db_to_amp(_denormalize(spectrogram, hparams) + hparams['ref_level_db'])  # dB 스펙트로그램을 선형 값으로 변환
    return inv_preemphasis(_griffin_lim(S ** hparams['power'], hparams), hparams['preemphasis'], hparams['preemphasize'])  # Griffin-Lim 알고리즘으로 위상 복원 후 Pre-emphasis 제거


def _lws_processor(hparams):
    # LWS (Local Weighted Sum) 처리기 생성
    import lws
    return lws.lws(hparams['fft_size'], get_hop_size(hparams), fftsize=hparams['win_size'], mode="speech")


def _griffin_lim(S, hparams):
    '''librosa의 Griffin-Lim 알고리즘 구현
    Griffin-Lim 알고리즘은 주어진 스펙트로그램에서 위상을 복원하여 음파를 생성
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))  # 랜덤 초기 위상 생성
    S_complex = np.abs(S).astype(np.complex)  # 복소수 형태로 변환
    y = _istft(S_complex * angles, hparams)  # 초기 위상으로 음파 생성
    for i in range(hparams['griffin_lim_iters']):  # 지정된 반복 횟수만큼 위상 복원 반복
        angles = np.exp(1j * np.angle(_stft(y, hparams)))  # 새로운 위상 계산
        y = _istft(S_complex * angles, hparams)  # 새 위상으로 음파 생성
    return y


def _stft(y, hparams):
    # Short-Time Fourier Transform (STFT)을 계산
    if hparams['use_lws']:
        return _lws_processor(hparams).stft(y).T  # LWS를 사용할 경우 LWS로 처리
    else:
        return librosa.stft(y=y, n_fft=hparams['fft_size'], hop_length=get_hop_size(hparams), win_length=hparams['win_size'])  # librosa로 STFT 수행


def _istft(y, hparams):
    # Inverse Short-Time Fourier Transform (iSTFT)을 계산
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams['win_size'])


def num_frames(length, fsize, fshift):
    """스펙트로그램의 프레임 개수를 계산
    length: 입력 신호의 길이
    fsize: 프레임 크기 (FFT 크기)
    fshift: 프레임 이동 크기 (Hop 크기)
    """
    pad = (fsize - fshift)  # 패딩 크기 계산
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """좌우 패딩 크기를 계산
    x: 입력 신호
    fsize: 프레임 크기 (FFT 크기)
    fshift: 프레임 이동 크기 (Hop 크기)
    """
    M = num_frames(len(x), fsize, fshift)  # 프레임 개수 계산
    pad = (fsize - fshift)  # 패딩 크기 계산
    T = len(x) + 2 * pad  # 패딩 적용 후 신호 길이
    r = (M - 1) * fshift + fsize - T  # 오른쪽 패딩 크기
    return pad, pad + r


def librosa_pad_lr(x, fsize, fshift):
    '''librosa에서 사용하는 우측 패딩 크기 계산
    x: 입력 신호
    fsize: 프레임 크기 (FFT 크기)
    fshift: 프레임 이동 크기 (Hop 크기)
    '''
    return int(fsize // 2)

# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram, hparams):
    # 선형 스펙트로그램을 멜 스펙트로그램으로 변환
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)  # 멜 필터 생성
    return np.dot(_mel_basis, spectogram)  # 필터 적용하여 변환


def _mel_to_linear(mel_spectrogram, hparams):
    # 멜 스펙트로그램을 선형 스펙트로그램으로 변환
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))  # 멜 필터의 역행렬 생성
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))  # 필터 적용하여 변환


def _build_mel_basis(hparams):
    # 멜 필터를 생성
    return librosa.filters.mel(sr=hparams['sample_rate'], n_fft=hparams['fft_size'], n_mels=hparams['num_mels'])


def _amp_to_db(x, hparams):
    # 진폭을 데시벨로 변환
    min_level = np.exp(hparams['min_level_db'] / 20 * np.log(10))  # 최소 dB 수준
    return 20 * np.log10(np.maximum(min_level, x))  # 데시벨 계산


def _db_to_amp(x):
    # 데시벨을 진폭으로 변환
    return np.power(10.0, (x) * 0.05)


def _normalize(S, hparams):
    # 스펙트로그램 정규화
    if hparams['allow_clipping_in_normalization']:
        if hparams['symmetric_mels']:
            return np.clip((2 * hparams['max_abs_value']) * ((S - hparams['min_level_db']) / (-hparams['min_level_db'])) - hparams['max_abs_value'],
                           -hparams['max_abs_value'], hparams['max_abs_value'])
        else:
            return np.clip(hparams['max_abs_value'] * ((S - hparams['min_level_db']) / (-hparams['min_level_db'])), 0, hparams['max_abs_value'])
    assert S.max() <= 0 and S.min() - hparams['min_level_db'] >= 0
    if hparams['symmetric_mels']:
        return (2 * hparams['max_abs_value']) * ((S - hparams['min_level_db']) / (-hparams['min_level_db'])) - hparams['max_abs_value']
    else:
        return hparams['max_abs_value'] * ((S - hparams['min_level_db']) / (-hparams['min_level_db']))

def _denormalize(D, hparams):
    # 정규화된 스펙트로그램을 원래 값으로 복원
    if hparams['allow_clipping_in_normalization']:
        # 정규화에서 클리핑이 허용된 경우
        if hparams['symmetric_mels']:
            # 대칭 정규화된 멜 스펙트로그램 처리
            return (((np.clip(D, -hparams['max_abs_value'], hparams['max_abs_value']) + hparams['max_abs_value'])
                     * -hparams['min_level_db'] / (2 * hparams['max_abs_value'])) + hparams['min_level_db'])
        else:
            # 비대칭 정규화된 멜 스펙트로그램 처리
            return ((np.clip(D, 0, hparams['max_abs_value']) * -hparams['min_level_db'] / hparams['max_abs_value'])
                    + hparams['min_level_db'])

    # 정규화에서 클리핑이 허용되지 않은 경우
    if hparams['symmetric_mels']:
        # 대칭 정규화된 멜 스펙트로그램 처리
        return (((D + hparams['max_abs_value']) * -hparams['min_level_db'] / (2 * hparams['max_abs_value'])) + hparams['min_level_db'])
    else:
        # 비대칭 정규화된 멜 스펙트로그램 처리
        return ((D * -hparams['min_level_db'] / hparams['max_abs_value']) + hparams['min_level_db'])


def mulaw(x, mu=256):
    """
    Mu-Law 압축 (Companding)
    입력 신호 x를 Mu-Law 방식으로 압축.
    Args:
        x: 입력 신호, [-1, 1] 범위의 값.
        mu: 압축 파라미터 μ.
    Returns:
        압축된 신호, [-1, 1] 범위의 값.
    """
    return _sign(x) * _log1p(mu * _abs(x)) / _log1p(mu)


def inv_mulaw(y, mu=256):
    """
    Mu-Law 압축 해제 (Expansion)
    압축된 신호 y를 복원.
    Args:
        y: 압축된 신호, [-1, 1] 범위의 값.
        mu: 압축 파라미터 μ.
    Returns:
        복원된 신호, [-1, 1] 범위의 값.
    """
    return _sign(y) * (1.0 / mu) * ((1.0 + mu)**_abs(y) - 1.0)


def mulaw_quantize(x, mu=256):
    """
    Mu-Law 압축 및 양자화 (Quantization)
    Args:
        x: 입력 신호, [-1, 1] 범위의 값.
        mu: 압축 파라미터 μ.
    Returns:
        양자화된 신호, [0, μ] 범위의 정수 값.
    """
    mu = mu - 1
    y = mulaw(x, mu)
    return _asint((y + 1) / 2 * mu)  # [-1, 1] 범위를 [0, μ]로 스케일링


def inv_mulaw_quantize(y, mu=256):
    """
    Mu-Law 압축 해제 및 양자화 복원
    Args:
        y: 양자화된 신호, [0, μ] 범위의 값.
        mu: 압축 파라미터 μ.
    Returns:
        복원된 신호, [-1, 1] 범위의 값.
    """
    mu = mu - 1
    y = 2 * _asfloat(y) / mu - 1  # [0, μ] 범위를 [-1, 1]로 변환
    return inv_mulaw(y, mu)


# TensorFlow와 NumPy 배열에 대해 작동하는 래퍼 함수들
def _sign(x):
    # 신호의 부호 계산
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.sign(x) if (isnumpy or isscalar) else tf.sign(x)


def _log1p(x):
    # log(1 + x) 계산
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.log1p(x) if (isnumpy or isscalar) else tf.log1p(x)


def _abs(x):
    # 절댓값 계산
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.abs(x) if (isnumpy or isscalar) else tf.abs(x)


def _asint(x):
    # 정수로 변환
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.int) if isnumpy else int(x) if isscalar else tf.cast(x, tf.int32)


def _asfloat(x):
    # float로 변환
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.float32) if isnumpy else float(x) if isscalar else tf.cast(x, tf.float32)

def frames_to_hours(n_frames, hparams):
    # 총 프레임 수를 시간 단위로 변환
    # n_frames: 프레임 수의 리스트
    # hparas.['frame_shift_ms']: 프레임 간 시간 간격 (밀리초)
    # 3600 * 1000으로 나눠 시간을 계산
    return sum((n_frame for n_frame in n_frames)) * hparams['frame_shift_ms'] / (3600 * 1000)

def get_duration(audio, hparams):
    # 오디오의 총 길이(초) 계산
    # librosa의 get_duration 메서드를 사용하여 샘플 레이트에 맞게 계산
    return librosa.core.get_duration(audio, sr=hparams['sample_rate'])

def _db_to_amp_tensorflow(x):
    # 데시벨 값을 선형 값으로 변환
    # x: 데시벨 값
    # tf.pow 함수로 10^(x * 0.05)를 계산
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _denormalize_tensorflow(S, hparams):
    # TensorFlow에서 정규화 해제
    # 정규화된 스펙트로그램을 원래 값으로 복원
    return (tf.clip_by_value(S, 0, 1) * -hparams['min_level_db']) + hparams['min_level_db']

def _griffin_lim_tensorflow(S, hparams):
    # Griffin-Lim 알고리즘을 TensorFlow로 구현
    # 위상 정보를 복원하여 파형 재구성
    with tf.variable_scope('griffinlim'):
        # 스펙트로그램 차원을 확장하여 배치 차원 추가
        S = tf.expand_dims(S, 0)
        # 복소수 형태의 스펙트로그램 초기화
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        # 초기 복원된 파형 생성
        y = _istft_tensorflow(S_complex, hparams)
        # Griffin-Lim 반복 수행
        for i in range(hparams['griffin_lim_iters']):
            est = _stft_tensorflow(y, hparams)  # STFT 재계산
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)  # 위상 계산
            y = _istft_tensorflow(S_complex * angles, hparams)  # ISTFT로 파형 재구성
        return tf.squeeze(y, 0)  # 배치 차원 제거 후 반환

def _istft_tensorflow(stfts, hparams):
    # TensorFlow에서 ISTFT 계산
    # stfts: 복소수 형태의 스펙트로그램
    # hparams로부터 FFT 크기, hop 크기, 윈도우 크기 계산
    n_fft, hop_length, win_length = _stft_parameters(hparams)
    return tf.signal.inverse_stft(stfts, win_length, hop_length, n_fft)

def _stft_tensorflow(signals, hparams):
    # TensorFlow에서 STFT 계산
    # signals: 입력 신호
    # hparams로부터 FFT 크기, hop 크기, 윈도우 크기 계산
    n_fft, hop_length, win_length = _stft_parameters(hparams)
    return tf.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)

def _stft_parameters(hparams):
    # STFT를 위한 파라미터 계산
    # n_fft: FFT 크기
    # hop_length: 프레임 간 이동 크기
    # win_length: 윈도우 크기
    n_fft = (hparams['num_freq'] - 1) * 2  # 주파수 개수를 기준으로 FFT 크기 계산
    hop_length = int(hparams['frame_shift_ms'] / 1000 * hparams['sample_rate'])  # 프레임 이동 크기 계산
    win_length = int(hparams['frame_length_ms'] / 1000 * hparams['sample_rate'])  # 윈도우 크기 계산
    return n_fft, hop_length, win_length