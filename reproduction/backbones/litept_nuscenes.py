_base_ = ["${PROJECT_ROOT}/sources/LitePT/configs/nuscenes/semseg-litept-small-v1m1.py"]
data_root = "${PROJECT_ROOT}/data/litept"
weight = "${PROJECT_ROOT}/checkpoints/LitePT-S/model_best.pth"
save_path = "${PROJECT_ROOT}/outputs/litept_val"
num_worker = 4
batch_size_test = 1
data = dict(
    train=dict(data_root=data_root),
    val=dict(data_root=data_root),
    test=dict(data_root=data_root, split="val"),
)
