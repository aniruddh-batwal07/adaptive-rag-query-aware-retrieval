"""Tests for router comparison infrastructure (Task 7).

These tests verify the router comparison framework works correctly
without modifying existing M11/M12 results or primary benchmark numbers.
"""

import pytest
import os
import json
import torch
from unittest.mock import patch, MagicMock
from src.router.query_classifier import QueryClassifier
from src.router.routing_logic import AdaptiveController, RoutingDecision
from src.utils.config import load_config, Config


class TestRouterSelection:
    """Tests for router model selection and loading."""

    def test_distilbert_router_loads_from_hub(self):
        """DistilBERT router can be initialized from HuggingFace hub name."""
        classifier = QueryClassifier("hf-internal-testing/tiny-random-distilbert", device="cpu")
        assert classifier.model is not None
        assert classifier.tokenizer is not None

    def test_bert_router_loads_from_hub(self):
        """BERT router can be initialized from HuggingFace hub name."""
        classifier = QueryClassifier("hf-internal-testing/tiny-random-bert", device="cpu")
        assert classifier.model is not None
        assert classifier.tokenizer is not None

    def test_classifier_output_format_distilbert(self):
        """DistilBERT classifier returns correct output format."""
        classifier = QueryClassifier("hf-internal-testing/tiny-random-distilbert", device="cpu")
        result = classifier.predict("What is the capital of France?")
        assert "complexity_label" in result
        assert "confidence" in result
        assert result["complexity_label"] in ["SIMPLE", "COMPLEX"]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_classifier_output_format_bert(self):
        """BERT classifier returns correct output format."""
        classifier = QueryClassifier("hf-internal-testing/tiny-random-bert", device="cpu")
        result = classifier.predict("What is the capital of France?")
        assert "complexity_label" in result
        assert "confidence" in result
        assert result["complexity_label"] in ["SIMPLE", "COMPLEX"]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_both_routers_same_labels(self):
        """Both router types produce the same label set."""
        distilbert = QueryClassifier("hf-internal-testing/tiny-random-distilbert", device="cpu")
        bert = QueryClassifier("hf-internal-testing/tiny-random-bert", device="cpu")
        assert distilbert.id2label == bert.id2label
        assert distilbert.label2id == bert.label2id

    def test_batch_prediction_consistency(self):
        """Batch prediction returns same number of results as inputs."""
        classifier = QueryClassifier("hf-internal-testing/tiny-random-bert", device="cpu")
        queries = ["Question one?", "Question two?", "Question three?"]
        results = classifier.predict_batch(queries)
        assert len(results) == 3
        for r in results:
            assert r["complexity_label"] in ["SIMPLE", "COMPLEX"]


class TestRouterConfigSelection:
    """Tests for config-based router selection."""

    def test_default_config_has_distilbert(self):
        """Default config specifies distilbert-base-uncased."""
        config = load_config("configs/config.yaml")
        assert config.models.router.name == "distilbert-base-uncased"

    def test_bert_config_exists_and_loads(self):
        """BERT router config file exists and loads correctly."""
        bert_config_path = "configs/config_bert_router.yaml"
        if os.path.exists(bert_config_path):
            config = load_config(bert_config_path)
            assert config.models.router.name == "bert-base-uncased"
            assert config.models.router.checkpoint == "models/router_bert"
        else:
            pytest.skip("BERT router config not yet created")

    def test_routing_logic_unchanged(self):
        """Routing policy is unchanged: SIMPLE->K=2, COMPLEX->K=10."""
        config = load_config("configs/config.yaml")
        controller = AdaptiveController(config)
        
        simple = controller.get_decision({"complexity_label": "SIMPLE", "confidence": 0.9})
        assert simple.complexity == "SIMPLE"
        assert simple.retrieval_k == 2
        assert simple.compression_enabled is False
        
        complex_d = controller.get_decision({"complexity_label": "COMPLEX", "confidence": 0.9})
        assert complex_d.complexity == "COMPLEX"
        assert complex_d.retrieval_k == 10
        assert complex_d.compression_enabled is True


class TestMetricGeneration:
    """Tests for metric computation."""

    def test_confusion_matrix_format(self):
        """Confusion matrix should be 2x2 for binary classification."""
        from sklearn.metrics import confusion_matrix
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 1, 1, 1, 0]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        assert cm.shape == (2, 2)
        # TN=1, FP=1, FN=1, TP=2
        assert cm[0][0] == 1
        assert cm[0][1] == 1
        assert cm[1][0] == 1
        assert cm[1][1] == 2

    def test_binary_label_encoding(self):
        """Label encoding is consistent: SIMPLE=0, COMPLEX=1."""
        labels = ["SIMPLE", "COMPLEX", "SIMPLE"]
        encoded = [1 if l == "COMPLEX" else 0 for l in labels]
        assert encoded == [0, 1, 0]


class TestDataFairness:
    """Tests verifying both classifiers see identical data."""

    def test_val_data_exists(self):
        """Validation data file exists."""
        assert os.path.exists("datasets/router/val.jsonl")

    def test_val_data_format(self):
        """Validation data has correct format."""
        with open("datasets/router/val.jsonl", 'r') as f:
            first = json.loads(f.readline())
        assert "query" in first
        assert "label" in first
        assert first["label"] in ["SIMPLE", "COMPLEX"]

    def test_val_data_not_in_test_benchmark(self):
        """Validation data is separate from end-to-end benchmark."""
        val_queries = set()
        with open("datasets/router/val.jsonl", 'r') as f:
            for line in f:
                item = json.loads(line)
                val_queries.add(item["query"].strip().lower())
        
        test_queries = set()
        test_file = "datasets/splits/test_mini_10.jsonl"
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    q = item.get("query", item.get("question", "")).strip().lower()
                    test_queries.add(q)
            
            overlap = val_queries & test_queries
            assert len(overlap) == 0, f"Data leakage: {len(overlap)} queries overlap between val and test"

    def test_train_val_no_overlap(self):
        """Training and validation data do not overlap."""
        train_queries = set()
        with open("datasets/router/train.jsonl", 'r') as f:
            for line in f:
                item = json.loads(line)
                train_queries.add(item["query"].strip().lower())
        
        val_queries = set()
        with open("datasets/router/val.jsonl", 'r') as f:
            for line in f:
                item = json.loads(line)
                val_queries.add(item["query"].strip().lower())
        
        overlap = train_queries & val_queries
        assert len(overlap) == 0, f"Data leakage: {len(overlap)} queries overlap between train and val"

    def test_existing_router_eval_untouched(self):
        """Existing router_eval results are preserved."""
        metrics_path = "results/router_eval/metrics.json"
        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            metrics = json.loads(f.read())
        # Verify the existing metrics match expected values
        assert abs(metrics["accuracy"] - 0.9932885906040269) < 1e-10
        assert metrics["sample_count"] == 298

    def test_primary_benchmark_untouched(self):
        """Primary benchmark results are preserved."""
        for baseline in ["adaptive", "always_compress", "fixed_rag", "llm_only"]:
            metrics_path = f"results/primary_benchmark/{baseline}/metrics.json"
            assert os.path.exists(metrics_path), f"Missing: {metrics_path}"

    def test_m12_results_untouched(self):
        """M12 ablation results are preserved."""
        assert os.path.exists("results/m12/ablation_aggregate.json")
        assert os.path.exists("results/m12/adaptive_analysis.json")
