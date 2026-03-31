import json
from unittest.mock import MagicMock, patch

import pytest

from anti_distill.config import Settings
from anti_distill.dataset import HoldoutDataset
from anti_distill.teacher import TeacherSampler


@pytest.fixture
def settings(tmp_path):
    return Settings(
        teacher_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        dataset_path=str(tmp_path / "holdout.jsonl"),
        dataset_name="gsm8k",
        dataset_config="main",
        dataset_split="train",
        num_samples=2,
    )


# --- TeacherSampler ---

@patch("anti_distill.teacher.pipeline")
def test_teacher_sample(mock_pipeline, settings):
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": [{"role": "assistant", "content": "A helpful answer."}]}]
    mock_pipeline.return_value = mock_pipe

    sampler = TeacherSampler(settings)
    result = sampler.sample("What is ML?")
    assert result == "A helpful answer."


@patch("anti_distill.teacher.pipeline")
def test_generate_dataset_respects_num_samples(mock_pipeline, settings):
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": [{"role": "assistant", "content": "answer"}]}]
    mock_pipeline.return_value = mock_pipe

    sampler = TeacherSampler(settings)
    samples = sampler.generate_dataset(["p1", "p2", "p3"])
    assert len(samples) == 2  # num_samples=2


@patch("anti_distill.teacher.get_loader")
@patch("anti_distill.teacher.pipeline")
def test_load_prompts_from_gsm8k(mock_pipeline, mock_get_loader, settings):
    mock_pipeline.return_value = MagicMock()
    mock_loader = MagicMock()
    mock_loader.load.return_value = ["What is 1+1?", "What is 2+2?"]
    mock_get_loader.return_value = mock_loader

    sampler = TeacherSampler(settings)
    prompts = sampler.load_prompts()
    assert prompts == ["What is 1+1?", "What is 2+2?"]
    mock_get_loader.assert_called_once_with("gsm8k", config="main", split="train")


# --- HoldoutDataset ---

def test_dataset_save_and_load(settings):
    samples = [{"prompt": "q1", "response": "a1"}, {"prompt": "q2", "response": "a2"}]
    ds = HoldoutDataset(settings.dataset_path)
    ds.save(samples)
    loaded = ds.load()
    assert loaded == samples


def test_dataset_to_hf_dataset(settings):
    samples = [{"prompt": "What is AI?", "response": "A field of CS."}]
    ds = HoldoutDataset(settings.dataset_path)
    ds.save(samples)
    hf = ds.to_hf_dataset()
    assert "text" in hf.column_names
    assert "### Instruction:" in hf[0]["text"]
    assert "### Response:" in hf[0]["text"]


# --- Settings ---

def test_settings_validate_raises_without_api_key():
    s = Settings(openai_api_key="")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        s.validate()


@pytest.fixture
def settings():
    return Settings(openai_api_key="test-key", model="gpt-4o-mini")


@patch("anti_distill.client.OpenAI")
def test_chat_appends_history(mock_openai, settings):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello!"
    mock_openai.return_value.chat.completions.create.return_value = mock_response

    client = LLMClient(settings)
    result = client.chat("Hi")

    assert result == "Hello!"
    assert client.history[0]["role"] == "system"
    assert client.history[1]["role"] == "user"
    assert client.history[2]["role"] == "assistant"


@patch("anti_distill.client.OpenAI")
def test_reset_clears_history(mock_openai, settings):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello!"
    mock_openai.return_value.chat.completions.create.return_value = mock_response

    client = LLMClient(settings)
    client.chat("Hi")
    client.reset()

    assert client.history == []


def test_settings_validate_raises_without_api_key():
    settings = Settings(openai_api_key="")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.validate()
