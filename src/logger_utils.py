import logging
import os
from functools import wraps


def setup_logger(
    level=logging.DEBUG,
    file_level=logging.DEBUG,       
    console_level=logging.INFO, 
    file_path="log/log_dir/train.log", # 기본주소
    format_str=None,
    to_console=True,

):
    """
    파일 핸들러와 포맷터를 포함한 logger 객체 생성 함수
    """
    logger = logging.getLogger()
    logger.setLevel(level) # 로거 자체가 허용할 최소 레벨
    logger.handlers.clear()
    formatter = logging.Formatter(
        format_str or "%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s: %(message)s"
    )

    # 중복 핸들러 방지
    if not logger.handlers:
        # 파일 핸들러
        if file_path:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(file_level) # 파일에 기록되는 최소레벨
            logger.addHandler(file_handler)

        # 콘솔 핸들러
        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(console_level) # 출력 대상별(파일/콘솔)의 필터링 기준
            logger.addHandler(console_handler)

    