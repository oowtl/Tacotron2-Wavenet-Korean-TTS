# -*- coding: utf-8 -*-

# TensorFlow 버전 변경으로 인한 수정
# # TensorFlow 및 NumPy 라이브러리 임포트
# import tensorflow.compat.v1 as tf  # TensorFlow는 딥러닝 모델을 정의하고 학습하는 데 사용됩니다.
# from tensorboard.plugins.hparams import api as hp
# import numpy as np  # NumPy는 수학 연산 및 배열 처리를 지원합니다.

# collections 및 namedtuple 임포트
from collections import namedtuple

# 하이퍼 파라미터 설정을 위해 namedtuple을 사용하여 tuple을 생성하는 함수
def get_hparams(**kwargs):
    return namedtuple('GenericDict', kwargs.keys())(**kwargs)

# Tacotron-2 하이퍼파라미터 설정
hparams = get_hparams(
    name='Tacotron-2',  # 모델 이름

    # Tacotron 관련 하이퍼파라미터
    cleaners='korean_cleaners',  # 텍스트 정리(cleaning) 방식 ('korean_cleaners' 또는 'english_cleaners')

    skip_path_filter=False,  # npz 파일에서 불필요한 데이터를 필터링할지 결정
    use_lws=False,  # Griffin-Lim 알고리즘 대신 LWS를 사용할지 여부

    # 오디오 설정
    sample_rate=24000,  # 오디오 샘플링 속도 (Hz)

    # FFT 및 스펙트로그램 관련 설정
    hop_size=300,  # FFT의 프레임 이동 크기
    fft_size=2048,  # FFT의 크기
    win_size=1200,  # 윈도우 크기 (ms 단위)
    num_mels=80,  # 멜 스펙트로그램의 필터 개수 (스펙트로그램 높이)

    # 스펙트로그램 전처리
    preemphasize=True,  # 신호에 프리엠퍼시스 필터를 적용할지 여부
    preemphasis=0.97,  # 프리엠퍼시스 계수
    min_level_db=-100,  # 최소 dB 값 (잡음 제거)
    ref_level_db=20,  # 기준 dB 값
    signal_normalization=True,  # 스펙트로그램 신호를 정규화할지 여부
    allow_clipping_in_normalization=True,  # 정규화 과정에서 클리핑을 허용할지 여부
    symmetric_mels=True,  # 데이터를 0을 중심으로 대칭적으로 스케일링할지 여부
    max_abs_value=4.0,  # 스펙트로그램의 최대 절대값 (스케일링 범위)

    rescaling=True,  # 오디오 데이터를 리샘플링할지 여부
    rescaling_max=0.999,  # 리샘플링 후 최대값

    trim_silence=True,  # 오디오 시작과 끝의 침묵을 제거할지 여부
    trim_fft_size=512,  # 침묵 제거를 위한 FFT 크기
    trim_hop_size=128,  # 침묵 제거를 위한 프레임 이동 크기
    trim_top_db=23,  # 침묵으로 간주할 dB 기준값

    # 메모리 초과 방지를 위한 설정
    clip_mels_length=True,  # 멜 스펙트로그램 길이를 제한할지 여부
    max_mel_frames=1000,  # 최대 멜 프레임 수

    # 모델 입력 및 정규화 관련 설정
    l2_regularization_strength=0,  # L2 정규화 계수
    sample_size=9000,  # 학습에 사용할 샘플 크기
    silence_threshold=0,  # 음소거로 간주할 볼륨 임계값

    filter_width=3,  # 필터 크기
    gc_channels=32,  # 글로벌 컨디셔닝 벡터의 차원
    input_type="raw",  # 입력 유형 ('raw', 'mulaw', 'mulaw-quantize')
    scalar_input=True,  # 입력이 스칼라 값인지 여부

    # WaveNet 모델 구조
    dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512] * 2,  # WaveNet의 dilation 레이어
    residual_channels=128,  # Residual 연결의 채널 크기
    dilation_channels=256,  # Dilation 레이어 채널 크기
    quantization_channels=256,  # 양자화 채널 크기
    out_channels=30,  # 출력 채널 크기 (3의 배수, Logistic Loss에 적합)
    skip_channels=128,  # Skip 연결의 채널 크기
    use_biases=True,  # 모델에 Bias를 사용할지 여부
    upsample_type='SubPixel',  # 업샘플링 방식 ('SubPixel' 또는 None)
    upsample_factor=[12, 25],  # 업샘플링 계수 (곱이 hop_size와 동일해야 함)

    # WaveNet 학습 설정
    wavenet_batch_size=2,  # 배치 크기
    store_metadata=False,  # 메타데이터 저장 여부
    num_steps=1000000,  # 학습 단계 수
    wavenet_learning_rate=1e-3,  # 초기 학습률
    wavenet_decay_rate=0.5,  # 학습률 감소 비율
    wavenet_decay_steps=300000,  # 학습률 감소 단계
    wavenet_clip_gradients=True,  # 그래디언트 클리핑 여부

    # WaveNet 안정성 향상을 위한 설정
    legacy=True,  # Skip 연결에서 sqrt(0.5)를 곱할지 여부
    residual_legacy=True,  # Residual 블록 출력에 sqrt(0.5)를 곱할지 여부
    wavenet_dropout=0.05,  # 드롭아웃 비율

    optimizer='adam',  # 최적화 알고리즘
    momentum=0.9,  # 모멘텀 값
    max_checkpoints=3,  # 유지할 체크포인트 개수

    ####################################
    # TACOTRON HYPERPARAMETERS
    # Tacotron 모델의 학습, 모델 구성, 평가에 관련된 하이퍼파라미터 설정

    # Training
    adam_beta1=0.9,  # Adam 옵티마이저의 첫 번째 모멘텀 계수
    adam_beta2=0.999,  # Adam 옵티마이저의 두 번째 모멘텀 계수

    # Learning rate schedule
    tacotron_decay_learning_rate=True,  # 학습률이 지수 감소 형태를 따를지 여부
    tacotron_start_decay=40000,  # 학습률 감소를 시작하는 단계 (스텝)
    tacotron_decay_steps=18000,  # 학습률 감소 간격 (단계 수)
    tacotron_decay_rate=0.5,  # 학습률 감소 비율
    tacotron_initial_learning_rate=1e-3,  # 초기 학습률
    tacotron_final_learning_rate=1e-4,  # 최소 학습률

    # 초기 데이터 처리 관련 설정
    initial_data_greedy=True,  # 초기 단계에서 데이터를 균등하게 분배할지 여부
    initial_phase_step=8000,  # 지정된 단계 이전에는 균등 분배
    main_data_greedy_factor=0,  # 주 데이터에 추가 가중치를 줄지 여부
    main_data=[''],  # 주요 데이터를 포함하는 디렉터리
    prioritize_loss=False,  # 손실 기반 데이터 우선순위를 사용할지 여부

    # Model
    model_type='multi-speaker',  # 모델 유형 ('single' 또는 'multi-speaker')
    speaker_embedding_size=16,  # 화자 임베딩 벡터 크기

    embedding_size=512,  # 텍스트 임베딩 크기
    dropout_prob=0.5,  # 드롭아웃 비율

    reduction_factor=2,  # 감축 계수, 작을수록 더 많은 반복 필요 (메모리 소모 증가)

    # Encoder
    enc_conv_num_layers=3,  # 인코더에서 사용하는 컨볼루션 레이어 수
    enc_conv_kernel_size=5,  # 컨볼루션 커널 크기
    enc_conv_channels=512,  # 각 컨볼루션 레이어의 채널 수
    tacotron_zoneout_rate=0.1,  # Zoneout 비율
    encoder_lstm_units=256,  # LSTM 레이어의 유닛 수

    # Attention
    attention_type='bah_mon_norm',  # Attention 메커니즘 유형 ('loc_sen', 'bah_mon_norm')
    attention_size=128,  # Attention 벡터 크기
    smoothing=False,  # Attention 정규화 함수에서 스무딩을 사용할지 여부
    attention_dim=128,  # Attention 공간의 차원
    attention_filters=32,  # Attention 컨볼루션 필터 수
    attention_kernel=(31,),  # Attention 컨볼루션 커널 크기
    cumulative_weights=True,  # 이전 Attention 가중치를 누적할지 여부

    # Attention synthesis constraints
    synthesis_constraint=False,  # 합성에서 Attention 제약 조건을 사용할지 여부
    synthesis_constraint_type='window',  # 제약 유형 ('window', 'monotonic')
    attention_win_size=7,  # Attention 창 크기 (양방향 크기)

    # Loss parameters
    mask_encoder=True,  # Encoder 패딩을 마스킹할지 여부

    # Decoder
    prenet_layers=[256, 256],  # 프리넷 레이어 구성
    decoder_layers=2,  # 디코더 LSTM 레이어 수
    decoder_lstm_units=1024,  # 디코더 LSTM의 유닛 수
    dec_prenet_sizes=[256, 256],  # 프리넷 크기

    # Residual postnet
    postnet_num_layers=5,  # 포스트넷 컨볼루션 레이어 수
    postnet_kernel_size=(5,),  # 포스트넷 컨볼루션 커널 크기
    postnet_channels=512,  # 각 포스트넷 레이어의 채널 수

    # Linear mel spectrogram
    post_bank_size=8,  # 포스트넷의 컨볼루션 뱅크 크기
    post_bank_channel_size=128,  # 포스트넷의 채널 크기
    post_maxpool_width=2,  # 최대 풀링 크기
    post_highway_depth=4,  # Highway 네트워크 깊이
    post_rnn_size=128,  # RNN 레이어 크기
    post_proj_sizes=[256, 80],  # 프로젝션 레이어 크기 (num_mels=80)
    post_proj_width=3,  # 프로젝션 레이어 커널 크기

    # Regularization
    tacotron_reg_weight=1e-6,  # L2 정규화 가중치
    inference_prenet_dropout=True,  # 추론 시 프리넷 드롭아웃 적용 여부

    # Eval
    min_tokens=30,  # 학습에 사용될 최소 토큰 수
    min_n_frame=30 * 5,  # 최소 프레임 수 (reduction_factor와 곱함)
    max_n_frame=200 * 5,  # 최대 프레임 수
    skip_inadequate=False,  # 기준에 맞지 않는 데이터를 스킵할지 여부

    griffin_lim_iters=60,  # Griffin-Lim 알고리즘 반복 횟수
    power=1.5,  # Griffin-Lim 알고리즘에서 사용하는 파워 계수

)

# # LWS(Local Weighted Sum)를 사용할 경우 추가 설정
# if hparams.use_lws:
#     hparams.sample_rate = 20480  # 샘플 레이트 재설정
#     hparams.hop_size = 256  # 프레임 이동 크기
#     hparams.fft_size = 2048  # FFT 크기
#     hparams.win_size = None  # 윈도우 크기
# else:

# LWS 사용안함
hparams_dict = hparams._asdict()
print(hparams_dict)
hparams_dict['num_freq'] = int(hparams_dict['fft_size'] / 2 + 1)  # 주파수 대역 개수 계산
hparams_dict['frame_shift_ms'] = hparams_dict['hop_size'] * 1000.0 / hparams_dict['sample_rate']  # 프레임 이동 시간 계산
hparams_dict['frame_length_ms'] = hparams_dict['win_size'] * 1000.0 / hparams_dict['sample_rate']  # 프레임 길이 계산

hparams = get_hparams(**hparams_dict)
print(hparams)

# 하이퍼파라미터를 문자열로 반환하는 함수 (디버그용)
def hparams_debug_string():
    # values = hparams.values()  # 하이퍼파라미터 값을 가져옴
    hp_fields = hparams._fields

    hp = ['  %s: %s' % (name, hparams[idx]) for [idx, name] in enumerate(sorted(hp_fields))]  # 정렬된 문자열 생성
    return 'Hyperparameters:\n' + '\n'.join(hp)  # 문자열 반환
