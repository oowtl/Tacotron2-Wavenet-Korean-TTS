# coding: utf-8
"""
스크립트 사용법
이 스크립트는 다중 화자 TTS 시스템의 데이터를 전처리하는 데 사용됩니다.
실행 예제
- python preprocess.py --num_workers 10 --name son --in_dir <입력 데이터 경로> --out_dir <출력 데이터 경로>
python preprocess.py --num_workers 8 --name son --in_dir .\/datasets\/son --out_dir .\/data\/son

- python preprocess.py --num_workers 10 --name moon --in_dir <입력 데이터 경로> --out_dir <출력 데이터 경로>
python preprocess.py --num_workers 8 --name moon --in_dir .\/datasets\/moon --out_dir .\/data\/moon

결과:
- out_dir에 다음과 같은 데이터를 포함하는 npz 파일이 생성됩니다:
- 'audio', 'mel', 'linear', 'time_steps', 'mel_frames', 'text', 'tokens', 'loss_coeff'
"""

# 필수 라이브러리 임포트
import argparse  # 명령줄 인자 처리
import os  # 파일 및 디렉토리 작업
from multiprocessing import cpu_count  # CPU 코어 개수 확인
from tqdm import tqdm  # 작업 진행 상황 표시
import importlib  # 동적 모듈 임포트
from hparams import hparams, hparams_debug_string  # 하이퍼파라미터 및 디버그 정보
import warnings  # 경고 메시지 제어

# FutureWarning 메시지 무시 설정
warnings.simplefilter(action='ignore', category=FutureWarning)


# 데이터 전처리 함수
def preprocess(mod, in_dir, out_root, num_workers):
    """
    전처리 과정을 실행하는 함수
    mod: 데이터셋 모듈
    in_dir: 입력 데이터 디렉토리
    out_root: 출력 데이터 루트 디렉토리
    num_workers: 병렬 작업에 사용할 워커 수
    """

    print("preprocess")

    os.makedirs(out_dir, exist_ok=True)  # 출력 디렉토리 생성 (이미 존재하면 무시)
    # 데이터 전처리 실행
    metadata = mod.build_from_path(hparams, in_dir, out_dir, num_workers=num_workers, tqdm=tqdm)
    # 메타데이터 저장
    write_metadata(metadata, out_dir)


# 메타데이터 저장 함수
def write_metadata(metadata, out_dir):
    """
    전처리 결과 메타데이터를 train.txt로 저장하고 요약 정보를 출력
    metadata: 전처리 결과 데이터
    out_dir: 출력 데이터 디렉토리
    """
    # train.txt 파일 생성
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')

    # 통계 정보 계산
    mel_frames = sum([int(m[4]) for m in metadata])  # 멜 프레임 총합
    timesteps = sum([int(m[3]) for m in metadata])  # 오디오 총 타임스텝
    sr = hparams['sample_rate']  # 샘플링 레이트
    hours = timesteps / sr / 3600  # 데이터 총 시간(시간 단위)

    # 통계 정보 출력
    print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
        len(metadata), mel_frames, timesteps, hours))
    print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))  # 최대 텍스트 길이
    print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))  # 최대 멜 프레임 길이
    print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))  # 최대 타임스텝 길이


if __name__ == "__main__":
    # 명령줄 인자 파서 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None, help="데이터셋 이름 (예: son, moon 등)")
    parser.add_argument('--in_dir', type=str, default=None, help="입력 데이터 디렉토리 경로")
    parser.add_argument('--out_dir', type=str, default=None, help="출력 데이터 디렉토리 경로")
    parser.add_argument('--num_workers', type=str, default=None, help="병렬 작업 워커 수")
    parser.add_argument('--hparams', type=str, default=None, help="하이퍼파라미터 문자열")
    args = parser.parse_args()

    # 하이퍼파라미터 적용
    if args.hparams is not None:
        hparams.parse(args.hparams)
    # print(hparams_debug_string())  # 하이퍼파라미터 디버그 정보 출력

    # 명령줄 인자 처리
    name = args.name  # 데이터셋 이름
    in_dir = args.in_dir  # 입력 데이터 디렉토리
    out_dir = args.out_dir  # 출력 데이터 디렉토리
    num_workers = args.num_workers  # 병렬 작업 워커 수
    num_workers = cpu_count() if num_workers is None else int(num_workers)  # 디폴트는 CPU 코어 개수

    print(hparams)
    print("Sampling frequency: {}".format(hparams['sample_rate']))  # 샘플링 레이트 출력

    # 데이터셋 이름 확인
    assert name in ["cmu_arctic", "ljspeech", "son", "moon"], "지원하지 않는 데이터셋 이름입니다."
    mod = importlib.import_module('datasets.{}'.format(name))  # 데이터셋 모듈 동적 임포트
    preprocess(mod, in_dir, out_dir, num_workers)  # 전처리 실행