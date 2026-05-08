import pytest
from mlplo.app import load_model_info, MODE_PRESETS
from pathlib import Path
import json

def test_load_model_info_fallback():
    info = load_model_info("facebook/bart-large-xsum")
    assert "Fallback Model" in info
    assert "facebook/bart-large-xsum" in info

def test_load_model_info_local(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    metrics_file = metrics_dir / "test_metrics.json"
    metrics_file.write_text(json.dumps({"test_rouge1": 0.45, "test_rougeL": 0.40}))
    
    info = load_model_info(str(tmp_path))
    assert "Local Checkpoint" in info
    assert "ROUGE-1" in info
    assert "0.45" in info

def test_mode_presets():
    assert "Quick Pulse" in MODE_PRESETS
    assert "max_new_tokens" in MODE_PRESETS["Quick Pulse"]
