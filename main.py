import torch
import mlflow
from src.dataset import Load_Dataset
from src.train import fit
from src.utils import setup_tensorboard,setup_mlflow,setup_logging,prepare_model,prepare_training_tools,create_optimizer,parse_args




def main():
    args = parse_args()
    writer = setup_tensorboard(args)
    setup_mlflow(args)
    setup_logging(args)
    train_dataset,train_dataloader, val_dataloader, test_dataloader = Load_Dataset(
            train_data_path=args.train_path,
            test_data_path=args.test_path,
            batch_size = args.batch_size
        )

    model = prepare_model()
    optimizer = create_optimizer(model,args)
    criterion,scheduler,early_stoper = prepare_training_tools(optimizer,args)
    # 5. MLflow 시작
    mlflow.set_experiment("digit-recognizer")
    with mlflow.start_run():
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("lr", args.lr)

        # 6. 학습 루프
        fit(model = model,
            train_dataloader = train_dataloader,
            val_dataloader = val_dataloader, 
            criterion = criterion,
            optim = optimizer,
            epochs = args.epochs,
            scheduler = scheduler,
            early_stopping = early_stoper,
            writer=writer)

        # 7. 모델 저장 or 추론
        torch.save(model.state_dict(), "model.pth")
    
if __name__ == "__main__":
    # trian시 argparse사용법
    # python main.py --config configs/train/exp1_best.json 실행.
    main()