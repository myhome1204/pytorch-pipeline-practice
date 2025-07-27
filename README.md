# pytorch-pipeline-practice

캐글의 고양이/강아지 이진 분류 예제를 기반으로 CNN 학습 파이프라인 구성 연습

단순 모델 구현에 그치지 않고, 하이퍼파라미터 탐색, 로깅, 시각화, 실험 추적 등 다양한 실험 관리 도구를 적용

결과 성능보다 ML 실무 파이프라인 흐름을 익히는 데 초점

캐글 정답 데이터가 없어 inference(예측) 모듈은 포함되지 않음

※ 로그 파일은 용량 및 수가 많아 Git 저장소에는 포함하지 않음

## 🔧 주요 기능 요약 (Features)

PyTorch 기반 CNN 모델 학습 구조 구현

Ray Tune을 통한 하이퍼파라미터 탐색 (lr, batch_size, weight_decay)

MLflow를 활용한 실험 기록 및 시각화

TensorBoard를 이용한 학습 곡선 및 gradient 흐름 시각화

EarlyStopping, Learning Rate Scheduler 적용

## 🛠 사용 기술 스택 (Tech Stack)

Framework: PyTorch, torchvision

Experiment Tracking: MLflow, TensorBoard

Hyperparameter Tuning: Ray Tune

Logging & Config: logging, argparse

Automation: Shell script (.sh)

## 📁 프로젝트 구조

```
├── main.py # (Train)실행 진입점
├── requirements.txt # 패키지 목록
├── README.md # 프로젝트 설명
├── .env # 환경변수 (로컬 경로 등)
├── .gitignore
├── checkpoint.pt # 모델 체크포인트
├── configs/ # 학습/튜닝 설정 파일
│ ├── train/
│ └── tune/
│
├── data/ # 데이터 디렉토리
│ ├── raw/ # 원본 데이터
│ └── processed/ # 전처리된 데이터
│
├── logs/ # 로그 관련 디렉토리
│ ├── logging/ # logging 모듈 로그
│ ├── mlflow/ # MLflow 실험 추적 로그
│ └── tensorboard/ # TensorBoard 로그
│
├── models/ # (저장모델)
│ ├── model.pth # 저장된 모델 파일
│
├── outputs/ # 실험 결과 저장
│ ├── experiments/ # 모델/결과 저장
│ ├── hparam/ # Ray Tune 튜닝 결과
│ └── inference_results/ # 예측 결과
│
├── scripts/ # 실행용 쉘 스크립트
│ ├── train/
│ └── tune/train_exp1_full.sh
│
└── src/ # 소스 코드
├── pycache/
├── dataset.py # 데이터 로딩 및 전처리
├── early_stopping.py # EarlyStopping 콜백
├── logger_utils.py # 커스텀 로깅 설정
├── model.py # CNN 모델 정의
├── train.py # 학습 로직
├── utils.py # 기타 유틸 함수
└── tune/ # Ray Tune 관련 코드(train_fn,run_tuner,tune_utils 등)
```

## ⚙️ 실행 방법 (Usage)

### 1️⃣ 하이퍼파라미터 탐색 (Ray Tune 사용)
configs/tune/search_full.yaml을 기반으로 lr, batch_size, weight_decay 등을 탐색
```
bash scripts/tune/train_exp1_full.sh
```
실행 시 내부적으로 다음 Python 명령이 실행
```
python src/tune/run_tuner.py \
  --name ray_exp1 \
  --config_file configs/tune/search_full.yaml \
  --num_samples 30 \
  --max_num_epochs 20 \
  --min_num_epochs 5 \
  --train_data_path "$TRAIN_DATA_PATH" \
  --gpus_per_trial 1 \
  --storage_path "$STORAGE_PATH" \
  --logging_path logs/logging/tune/ray_exp1.log \
  --logging_basic_level DEBUG \
  --logging_console_level INFO \
  --logging_file_level INFO \
  --save_path outputs/hparam
```

### 2️⃣ 모델 학습 (main.py + JSON config 사용)

선택된 하이퍼파라미터를 기반으로 실제 학습을 수행
config는 configs/train/train_config.json에 정의
```
python main.py --config_file configs/train/train_config.json
```
실행 시 내부적으로 다음 Python 명령이 실행

```
{
  "train_path": "data/raw/train.csv",
  "test_path" : "data/raw/test.csv",
  "lr": 0.00046,
  "batch_size": 64,
  "optimizer": "adam",
  "weight_decay": 0.00044,
  "logging_path": "logs/logging/train/train2.log",
  "logging_basic_level": "DEBUG",
  "logging_console_level": "INFO",
  "logging_file_level": "INFO",
  "mlflow_log_path": "logs/mlflow/train",
  "tensorboard_log_path": "logs/tensorboard/train/train2",
  "scheduler_patience": 3,
  "scheduler_lr_factor": 0.5,
  "scheduler_mode": "min",
  "earlystop_patience": 3,
  "earlystop_delta": 0.001,
  "earlystop_mode": "min"
}
```


