import os
import json
import logging
import argparse  # 新增：导入命令行参数解析模块
# import cnocr
from Dataset.dataset import OcrDataModule
from utils.utils import (
    set_logger,
    load_model_params,
    check_model_name,
    save_img,
    read_img,
    read_charset,
)
from utils.trainer import PlTrainer
from utils.utils import gen_model
from utils.transforms import (
    train_transform,
    ft_transform,
    test_transform,
)

logger = set_logger(log_level=logging.INFO)


def train(
        rec_model_name,
        index_dir,
        train_config_fp,
        finetuning,
        resume_from_checkpoint,
        pretrained_model_fp,
):
    """训练识别模型"""
    check_model_name(rec_model_name)
    train_config = json.load(open(train_config_fp))
    val_transform = test_transform
    data_mod = OcrDataModule(
        index_dir=index_dir,
        vocab_fp=train_config['vocab_fp'],
        img_folder=train_config['img_folder'],
        batch_size=train_config['batch_size'],
        train_transforms=train_transform if not finetuning else ft_transform,
        val_transforms=val_transform,
        train_bucket_size=train_config.get('train_bucket_size'),
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory'],
    )
    data_mod.setup('')

    trainer = PlTrainer(
        train_config, ckpt_fn=['Euclan', rec_model_name]
    )
    model = gen_model(rec_model_name, data_mod.vocab)
    logger.info(model)

    if pretrained_model_fp is not None:
        load_model_params(model, pretrained_model_fp)

    trainer.fit(
        model, datamodule=data_mod, resume_from_checkpoint=resume_from_checkpoint
    )


def visualize_example(example, fp_prefix):
    if not os.path.exists(os.path.dirname(fp_prefix)):
        os.makedirs(os.path.dirname(fp_prefix))
    image = example
    save_img(image, '%s-image.jpg' % fp_prefix)


def main():
    # 新增：解析命令行参数
    parser = argparse.ArgumentParser(description='训练OCR识别模型')
    # 模型相关参数
    parser.add_argument('--encoder_name', type=str, default='EULCAN_lite_136', help='编码器名称')
    parser.add_argument('--decoder_name', type=str, default='fc', help='解码器名称')
    # 路径相关参数
    parser.add_argument('--index_dir', type=str, default='./test', help='数据集索引目录')
    parser.add_argument('--train_config_fp', type=str, default='./train_config_gpu.json',
                        help='训练配置文件路径')
    parser.add_argument('--pretrained_model_fp', type=str, default=None, help='预训练模型路径（None则不加载）')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='恢复训练的检查点路径')
    # 训练模式参数
    parser.add_argument('--finetuning', action='store_true', default=True, help='是否微调（默认True）')
    parser.add_argument('--no_finetuning', action='store_false', dest='finetuning', help='关闭微调模式')

    args = parser.parse_args()  # 解析参数

    # 组装模型名称
    MODEL_NAME = f"{args.encoder_name}-{args.decoder_name}"

    # 调用训练函数（使用命令行参数）
    train(
        rec_model_name=MODEL_NAME,
        index_dir=args.index_dir,
        train_config_fp=args.train_config_fp,
        finetuning=args.finetuning,
        resume_from_checkpoint=args.resume_from_checkpoint,
        pretrained_model_fp=args.pretrained_model_fp,
    )


if __name__ == '__main__':
    main()
