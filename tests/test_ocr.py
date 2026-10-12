import numpy as np
from PIL import Image, ImageDraw
import pytest

from pipelines.ocr_pipeline import OCRPipeline


@pytest.fixture(scope="module")
def sample_ocr_image():
    """Creates a synthetic white image containing clear black text."""
    image = Image.new("RGB", (300, 100), color="white")
    draw = ImageDraw.Draw(image)
    # Draws simple text in the middle
    draw.text((10, 35), "TEST OCR", fill="black")
    return image


def test_ocr_pipeline_init():
    pipeline = OCRPipeline(device="cpu")
    assert pipeline is not None
    assert pipeline.device == "cpu"


def test_ocr_extract_text(sample_ocr_image):
    pipeline = OCRPipeline(device="cpu")
    res = pipeline.extract_text(sample_ocr_image)

    assert isinstance(res, dict)
    assert res.get("success") is True
    assert "text" in res
    # Verifies that some OCR text was captured
    assert isinstance(res["text"], str)


def test_ocr_extract_layout_data(sample_ocr_image):
    pipeline = OCRPipeline(device="cpu")
    res = pipeline.extract_layout_data(sample_ocr_image)

    assert isinstance(res, dict)
    assert res.get("success") is True
    assert "blocks" in res
    assert "raw_text" in res
    assert isinstance(res["blocks"], list)