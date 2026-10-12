import pytest
from inference.text_inference import TextInferenceEngine
from inference.qa_inference import QAInferenceEngine


@pytest.fixture(scope="module")
def text_engine():
    return TextInferenceEngine(device="cpu")


@pytest.fixture(scope="module")
def qa_engine():
    return QAInferenceEngine(model_name="google/flan-t5-small", device="cpu")


def test_summarization_valid_text(text_engine):
    long_text = (
        "Artificial intelligence is transforming industries across the globe. "
        "It assists doctors in identifying diseases from medical imaging, allows vehicles to navigate roads autonomously, "
        "and optimizes supply chains to prevent delays. Many businesses rely heavily on these advanced algorithmic tools "
        "to gain deeper insights from customer interactions and improve daily productivity."
    )
    result = text_engine.summarize_text(long_text, max_length=50, min_length=10)

    assert isinstance(result, dict)
    assert result.get("status") == "success"
    assert "summary" in result
    assert len(result["summary"]) > 0


def test_summarization_empty_text(text_engine):
    result = text_engine.summarize_text("")
    assert result.get("status") == "error"


def test_qa_answer_question(qa_engine):
    context = "The router provides internet to all computers and phones in the house."
    question = "What device provides internet?"
    result = qa_engine.answer_question(question, context)

    assert isinstance(result, dict)
    assert result.get("status") == "success"
    assert "answer" in result
    assert "router" in result["answer"].lower()


def test_qa_empty_inputs(qa_engine):
    res1 = qa_engine.answer_question("", "Some context")
    assert res1.get("status") == "error"

    res2 = qa_engine.answer_question("Where is it?", "")
    assert res2.get("status") == "error"