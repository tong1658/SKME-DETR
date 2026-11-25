import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'SKME_DETR.yaml'


def main():
    parser = argparse.ArgumentParser(description='Train the model defined in the accompanying YAML file.')
    parser.add_argument('--data', type=Path, default=ROOT / 'dataset/data.yaml')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--device', help='CUDA device index (e.g. 0) or cpu')
    parser.add_argument('--project', type=Path, default=ROOT / 'runs/train')
    parser.add_argument('--name', default=MODEL.stem)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    from ultralytics import RTDETR

    model = RTDETR(str(MODEL))
    model.train(
        data=str(args.data.resolve()),
        pretrained=False,
        cache=False,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
    )


if __name__ == '__main__':
    main()
