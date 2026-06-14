from Finance_RAG.parsers.resolved_json import ResolvedJsonParser

__all__ = ["PaddleOcrApiParser", "ResolvedJsonParser"]


def __getattr__(name):
    if name == "PaddleOcrApiParser":
        from Finance_RAG.parsers.paddle_ocr_api import PaddleOcrApiParser

        return PaddleOcrApiParser
    raise AttributeError(name)
