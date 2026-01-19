from cn_ocr import CnOcr
from PIL import Image
import numpy as np


def ocr_single_image_test(
        image_path,
        rec_model_fp=None,
        rec_model_backend='pytorch',
        rec_vocab_fp=None,
        context='gpu',
        rec_model_name='EULCAN_lite_136-fc',
        det_model_name='naive_det'
):
    """
    单张图片的OCR识别测试函数（仅输出识别结果，不做准确率对比）

    参数:
        image_path (str): 待识别的单张图片路径（支持jpg/png等格式）
        rec_model_fp (str, optional): 识别模型权重文件路径，默认None使用内置模型
        rec_model_backend (str): 识别模型后端，默认'pytorch'
        rec_vocab_fp (str, optional): 识别模型词汇表文件路径，默认None使用内置词汇表
        context (str): 运行上下文，'gpu'或'cpu'，默认'gpu'
        rec_model_name (str): 识别模型名称，默认'densenet_lite_136-fc'
        det_model_name (str): 检测模型名称，默认'naive_det'

    返回:
        str: 识别出的文本（若识别失败返回空字符串）
    """
    # 初始化OCR引擎
    ocr_kwargs = {
        'rec_model_backend': rec_model_backend,
        'context': context,
        'rec_model_name': rec_model_name,
        'det_model_name': det_model_name
    }
    if rec_model_fp is not None:
        ocr_kwargs['rec_model_fp'] = rec_model_fp
    if rec_vocab_fp is not None:
        ocr_kwargs['rec_vocab_fp'] = rec_vocab_fp

    ocr = CnOcr(**ocr_kwargs)

    # 校验图片路径有效性并读取图片
    try:
        # 读取图片（兼容PIL Image或numpy数组）
        image = Image.open(image_path)
    except Exception as e:
        print(f"图片读取失败: {e}")
        return ""

    # 执行OCR识别
    try:
        ocr_result = ocr.ocr(image)
    except Exception as e:
        print(f"OCR识别过程出错: {e}")
        return ""

    # 提取识别文本（假设仅一个文本块）
    if ocr_result and len(ocr_result) > 0 and 'text' in ocr_result[0]:
        recognized_text = ocr_result[0]['text'].strip()
    else:
        recognized_text = ""

    # 仅输出识别结果
    print("=" * 50)
    print(f"待识别图片: {image_path}")
    print(f"OCR识别结果: {recognized_text if recognized_text else '未识别到任何文本'}")
    print("=" * 50)

    return recognized_text


# 示例调用（单张图片测试）
if __name__ == "__main__":
    # 配置参数（根据实际情况修改）
    TEST_IMAGE_PATH = 'test_images/1.jpg'  # 单张测试图片路径
    REC_MODEL_FP = './eulcan.ckpt'
    REC_VOCAB_FP = './label_number.txt'

    # 调用单张图片测试函数
    result = ocr_single_image_test(
        image_path=TEST_IMAGE_PATH,
        rec_model_fp=REC_MODEL_FP,
        rec_vocab_fp=REC_VOCAB_FP,
        context='gpu',
        rec_model_name='EULCAN_lite_136-fc',
        det_model_name='naive_det'
    )
