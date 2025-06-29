import os
import logging
import sentencepiece as spm
import torch


def chinese_tokenizer_load():
    """加载中文分词器"""
    sp_chn = spm.SentencePieceProcessor()
    sp_chn.Load('{}.model'.format("../tokenizer/chn"))
    return sp_chn


def english_tokenizer_load():
    """加载英文分词器"""
    sp_eng = spm.SentencePieceProcessor()
    sp_eng.Load('{}.model'.format("../tokenizer/eng"))
    return sp_eng


def set_logger(log_path):
    """设置日志"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 文件处理器
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter("%(asctime)s - %(message)s")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler) 