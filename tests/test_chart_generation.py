"""Tests for scripts/generate_charts.py helper functions.

Covers:
    - _save produces SVG (not PNG)
    - _load_csv parses numeric and boolean fields
    - _load_csv handles missing file gracefully
    - _load_json returns dict or empty on missing
    - parse_args defaults
    - Colour constants are valid hex
"""

import csv
import json
import os
import tempfile

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.generate_charts import (
    _save,
    _load_csv,
    _load_json,
    parse_args,
    C_DQN,
    C_PPO,
    C_STATIC,
    SCENARIOS_KEY,
    SCENARIOS_DISPLAY,
)


# ---------------------------------------------------------------------------
# _save
# ---------------------------------------------------------------------------

class TestSave:
    """_save writes SVG only."""

    def test_saves_svg_file(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _save(fig, "test_chart", tmpdir)
            assert result.endswith(".svg")
            assert os.path.isfile(result)

    def test_does_not_save_png(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        with tempfile.TemporaryDirectory() as tmpdir:
            _save(fig, "test_chart", tmpdir)
            png_path = os.path.join(tmpdir, "test_chart.png")
            assert not os.path.exists(png_path)

    def test_creates_directory_if_missing(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "sub", "dir")
            result = _save(fig, "chart", nested)
            assert os.path.isfile(result)

    def test_closes_figure(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig_num = fig.number
        with tempfile.TemporaryDirectory() as tmpdir:
            _save(fig, "chart", tmpdir)
        assert fig_num not in plt.get_fignums()


# ---------------------------------------------------------------------------
# _load_csv
# ---------------------------------------------------------------------------

class TestLoadCsv:
    """_load_csv parses CSV with type coercion."""

    def test_numeric_conversion(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["name", "value", "rate"])
            writer.writerow(["dqn", "42", "0.95"])
            writer.writerow(["ppo", "38", "0.91"])
            path = f.name
        try:
            rows = _load_csv(path)
            assert len(rows) == 2
            assert rows[0]["value"] == 42.0
            assert rows[0]["rate"] == 0.95
            assert rows[0]["name"] == "dqn"
        finally:
            os.unlink(path)

    def test_boolean_conversion(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["name", "enabled"])
            writer.writerow(["test", "true"])
            writer.writerow(["test2", "false"])
            path = f.name
        try:
            rows = _load_csv(path)
            assert rows[0]["enabled"] is True
            assert rows[1]["enabled"] is False
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty(self):
        rows = _load_csv("/nonexistent/path.csv")
        assert rows == []

    def test_empty_csv_returns_empty(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["col1", "col2"])
            path = f.name
        try:
            rows = _load_csv(path)
            assert rows == []
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# _load_json
# ---------------------------------------------------------------------------

class TestLoadJson:
    """_load_json reads JSON or returns empty dict."""

    def test_loads_valid_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"key": "value", "count": 5}, f)
            path = f.name
        try:
            data = _load_json(path)
            assert data["key"] == "value"
            assert data["count"] == 5
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty(self):
        data = _load_json("/nonexistent/file.json")
        assert data == {}


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestChartParseArgs:
    """parse_args returns correct defaults."""

    def test_defaults_contain_charts_dir(self):
        from unittest.mock import patch
        with patch("sys.argv", ["generate_charts.py"]):
            args = parse_args()
        assert "charts" in args.charts_dir
        assert "sprint7" in args.charts_dir


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Chart constants are valid."""

    def test_colour_hex_format(self):
        for colour in (C_DQN, C_PPO, C_STATIC):
            assert colour.startswith("#")
            assert len(colour) == 7

    def test_scenario_lists_match(self):
        assert len(SCENARIOS_KEY) == len(SCENARIOS_DISPLAY)
        assert len(SCENARIOS_KEY) == 4
