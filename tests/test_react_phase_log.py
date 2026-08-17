"""ReAct 步间阶段日志。"""

from __future__ import annotations

import json
from pathlib import Path

from llgraph.session.session_run_log import log_react_phase, run_log_path


def test_log_react_phase_appends_to_run_log(tmp_path: Path) -> None:
    log_react_phase(
        tmp_path,
        "cli-test",
        phase="compress_llm_start",
        detail={"tokens_before": 120000},
        duration_sec=1.5,
    )
    path = run_log_path(tmp_path, "cli-test")
    assert path.is_file()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[-1])
    assert record["event"] == "react_phase"
    assert record["phase"] == "compress_llm_start"
    assert record["tokens_before"] == 120000
