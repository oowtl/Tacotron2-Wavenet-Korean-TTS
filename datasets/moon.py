# -*- coding: utf-8 -*-
"""
데이터 전처리 스크립트:
- JSON 파일을 읽어 텍스트와 오디오 파일의 경로를 가져오고,
- 오디오 신호를 처리하여 Mel 및 Linear 스펙트로그램을 생성,
- 처리된 데이터를 npz 형식으로 저장하며,
- 학습을 위한 메타데이터를 반환합니다.
"""

# 필요한 라이브러리 임포트
from concurrent.futures import ProcessPoolExecutor  # 병렬 처리를 위한 프로세스 풀
from functools import partial  # 함수의 일부 인자를 고정하는 유틸리티
import numpy as np  # 수학적 연산 및 배열 처리
import os, json  # 파일 및 JSON 처리
from utils import audio  # 오디오 처리 유틸리티
from text import text_to_sequence  # 텍스트를 시퀀스로 변환하는 함수


# 데이터 전처리 메인 함수
def build_from_path(hparams, in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    """
    주어진 입력 데이터 경로에서 음성 데이터를 전처리하여 출력 디렉토리에 저장합니다.

    Args:
        - hparams: 하이퍼파라미터 객체
        - in_dir: 입력 데이터 디렉토리
        - out_dir: 처리된 데이터를 저장할 출력 디렉토리
        - num_workers: 병렬 작업에 사용할 워커 수 (기본값: 1)
        - tqdm: 진행 상황 표시를 위한 tqdm 함수

    Returns:
        - 학습 예제를 설명하는 튜플 리스트 (train.txt로 저장 가능)
    """
    executor = ProcessPoolExecutor(max_workers=num_workers)  # 병렬 작업을 위한 프로세스 풀 생성
    futures = []  # 작업을 저장할 리스트
    index = 1  # 작업 인덱스 초기화

    # JSON 파일 경로 설정
    path = os.path.join(in_dir, 'moon-recognition-All.json')

    # JSON 파일 읽기
    with open(path, encoding='utf-8') as f:
        content = f.read()
        data = json.loads(content)  # JSON 데이터를 딕셔너리로 변환

        # 각 텍스트와 오디오 파일에 대해 작업 추가
        for key, text in data.items():
            wav_path = key.strip().split('/')  # 오디오 경로 추출
            wav_path = os.path.join(in_dir, 'audio', '%s' % wav_path[-1])

            # 오디오 파일이 존재하지 않는 경우 스킵
            if not os.path.exists(wav_path):
                continue

            # 병렬 작업으로 _process_utterance 함수 호출
            futures.append(executor.submit(partial(_process_utterance, out_dir, wav_path, text, hparams)))
            index += 1

    # 작업 결과를 리스트로 반환
    return [future.result() for future in tqdm(futures) if future.result() is not None]


# 단일 발화 데이터를 처리하는 함수
def _process_utterance(out_dir, wav_path, text, hparams):
    """
    단일 발화 데이터 (오디오와 텍스트 쌍)를 전처리합니다.

    - Mel 및 Linear 스펙트로그램을 생성하고,
    - 처리된 데이터를 npz 형식으로 저장하며,
    - 학습에 필요한 메타데이터를 반환합니다.

    Args:
        - out_dir: 처리된 데이터를 저장할 출력 디렉토리
        - wav_path: 입력 오디오 파일 경로
        - text: 오디오와 연관된 텍스트
        - hparams: 하이퍼파라미터 객체

    Returns:
        - 학습 예제를 설명하는 튜플 (오디오 파일, Mel 스펙트로그램, Linear 스펙트로그램 등)
    """
    try:
        # 오디오 파일 로드
        wav = audio.load_wav(wav_path, sr=hparams['sample_rate'])
    except FileNotFoundError:  # 파일이 없을 경우 처리
        print(f'파일 {wav_path}이 존재하지 않습니다. 스킵합니다.')
        return None

    # 오디오 재스케일링
    if hparams['rescaling']:
        wav = wav / np.abs(wav).max() * hparams['rescaling_max']

    # 선행/후행 공백 제거 (필요할 경우)
    if hparams['trim_silence']:
        wav = audio.trim_silence(wav, hparams)

    # 오디오 데이터 인코딩 (mu-law, raw 등)
    if hparams['input_type'] == 'mulaw-quantize':
        out = audio.mulaw_quantize(wav, hparams['quantize_channels'])
        start, end = audio.start_and_end_indices(out, hparams['silence_threshold'])
        wav = wav[start:end]
        out = out[start:end]
        constant_values = audio.mulaw_quantize(0, hparams['quantize_channels'])
        out_dtype = np.int16
    elif hparams['input_type'] == 'mulaw':
        out = audio.mulaw(wav, hparams['quantize_channels'])
        constant_values = audio.mulaw(0., hparams['quantize_channels'])
        out_dtype = np.float32
    else:  # raw
        out = wav
        constant_values = 0.
        out_dtype = np.float32

    # Mel 스펙트로그램 생성
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    # 최대 프레임 길이 초과 시 None 반환
    if mel_frames > hparams['max_mel_frames'] and hparams['clip_mels_length']:
        return None

    # Linear 스펙트로그램 생성
    linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
    linear_frames = linear_spectrogram.shape[1]
    assert linear_frames == mel_frames  # Mel과 Linear 프레임 일치 확인

    # 오디오 패딩
    # LWS 호환 문제로 인한 사용 금지 처리
    # if hparas.['use_lws']:
    #     fft_size = hparas.['fft_size'] if hparas.['win_size'] is None else hparas.['win_size']
    #     l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))
    #     out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
    # else:
    pad = audio.librosa_pad_lr(wav, hparams['fft_size'], audio.get_hop_size(hparams))
    out = np.pad(out, pad, mode='reflect')

    # 오디오 데이터 길이 조정
    out = out[:mel_frames * audio.get_hop_size(hparams)]
    time_steps = len(out)

    # 처리된 데이터 저장
    wav_id = os.path.splitext(os.path.basename(wav_path))[0]
    npz_filename = f'{wav_id}.npz'
    data = {
        'audio': out.astype(out_dtype),
        'mel': mel_spectrogram.T,
        'linear': linear_spectrogram.T,
        'time_steps': time_steps,
        'mel_frames': mel_frames,
        'text': text,
        'tokens': text_to_sequence(text),
        'loss_coeff': 1
    }
    np.savez(os.path.join(out_dir, npz_filename), **data, allow_pickle=False)

    # 학습 예제를 설명하는 튜플 반환
    return (
    f'{wav_id}-audio.npy', f'{wav_id}-mel.npy', f'{wav_id}-linear.npy', time_steps, mel_frames, text, npz_filename)