# EULCAN
Efficient Ultra-Lightweight Convolutional Attention Network for Embedded Identity Document Recognition System

python >= 3.9


cd Eulcan


pip install requirment.txt



训练代码
python train.py \
  --encoder_name EULCAN_lite_136 \
  --decoder_name fc \
  --index_dir ./my_dataset/index \
  --train_config_fp ./my_train_config.json \
  --pretrained_model_fp ./pretrained/EULCAN_lite_136.pth


python test.py --image_path ./my_test.jpg --context cpu
