from PIL import Image, ImageDraw
import pytest
from inference.vision_inference import VisionInferenceEngine


@pytest.fixture(scope="module")
def vision_engine():
    return VisionInferenceEngine(device="cpu")


@pytest.fixture(scope="module")
def dummy_image():
    # Create an image with clear visual contrast (e.g., white canvas with colored shapes)
    img = Image.new("RGB", (224, 224), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 184, 184], fill="blue")
    return img


def test_vision_engine_init(vision_engine):
    assert vision_engine is not None
    assert vision_engine.device == "cpu"


def test_caption_generation(vision_engine, dummy_image):
    result = vision_engine.generate_caption(dummy_image, max_tokens=20)
    assert isinstance(result, dict)
    assert result.get("status") == "success"
    assert "caption" in result
    assert isinstance(result["caption"], str)
    assert len(result["caption"].strip()) > 0


def test_vqa_inference(vision_engine, dummy_image):
    result = vision_engine.answer_question(dummy_image, "What is shown in the image?")
    assert isinstance(result, dict)
    assert result.get("status") == "success"
    assert "answer" in result
    assert isinstance(result["answer"], str)
    # Validate structural correctness rather than semantic certainty on synthetic data
    assert len(result["answer"].strip()) > 0