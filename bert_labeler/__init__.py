"""BERT/ONNX 生产推理预留接口。第一阶段不包含训练或具体模型。"""

from typing import Protocol


class BertPredictor(Protocol):
    model_version: str

    def predict(self, model_inputs: list[str]) -> list[dict[str, object]]: ...


__all__ = ["BertPredictor"]
