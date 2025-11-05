from pathlib import Path

ROOT = Path(__file__).parent


class Config:
    # general
    seed = 42
    device = "auto"  # "cuda", "cpu", or "auto"

    # paths
    data_root = ROOT / "data"
    train_dir = data_root / "train"
    val_dir = data_root / "test"
    eval_dir = data_root / "eval"

    checkpoint_dir = ROOT / "checkpoints"
    resume_checkpoint = None  # set to a specific .pth if needed

    # data
    image_size = (512, 512)   # (W, H)
    batch_size = 2
    num_workers = 4
    shuffle_train = True
    shuffle_val = False

    # model
    num_classes = 2

    # training
    epochs = 50
    lr = 1e-4
    weight_decay = 0.0
    save_every = 5

    # metrics
    pixel_threshold = 0.5

    # outputs
    outputs_root = ROOT / "outputs"
    pred_masks_dir = outputs_root / "raw_masks"
    postproc_masks_dir = outputs_root / "postprocessed_masks"
