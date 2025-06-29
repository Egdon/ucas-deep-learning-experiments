import os
import logging
import sentencepiece as spm


def chinese_tokenizer_load():
    """加载中文SentencePiece分词器"""
    sp_chn = spm.SentencePieceProcessor()
    sp_chn.Load('{}.model'.format("./tokenizer/chn"))
    return sp_chn


def english_tokenizer_load():
    """加载英文SentencePiece分词器"""
    sp_eng = spm.SentencePieceProcessor()
    sp_eng.Load('{}.model'.format("./tokenizer/eng"))
    return sp_eng


def set_logger(log_path):
    """设置日志记录器，同时输出到终端和文件
    
    Args:
        log_path: 日志文件路径
    """
    if os.path.exists(log_path):
        os.remove(log_path)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 文件日志处理器
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # 控制台日志处理器
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler) 