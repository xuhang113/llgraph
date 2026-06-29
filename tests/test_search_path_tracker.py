"""search_path_tracker / 批量 grep harness 单测。"""

from __future__ import annotations

from llgraph.core.search_path_tracker import (
    expand_table_search_terms,
    extract_grep_batch_terms,
    format_retrieval_batch_hint,
)


def test_expand_table_search_terms() -> None:
    terms = expand_table_search_terms("tb_goods_describe_log")
    assert "tb_goods_describe_log" in terms
    assert "GoodsDescribeLog" in terms
    assert "GoodsDescribeLogDO" in terms
    assert "GoodsDescribeLogMapper" in terms


def test_extract_terms_from_table_question() -> None:
    msg = "tb_goods_describe_log 这个表是做什么用的，biz_id有重复，有什么影响"
    terms = extract_grep_batch_terms(msg)
    assert "tb_goods_describe_log" in terms
    assert "goodsdescribelogdo" in {t.lower() for t in terms if t.isascii()}
    assert "biz_id" in terms


def test_format_batch_hint_for_table_question() -> None:
    hint = format_retrieval_batch_hint(
        "tb_goods_describe_log 这个表是做什么用的，biz_id有重复"
    )
    assert "检索批量化" in hint
    assert "tb_goods_describe_log" in hint
    assert "禁止" in hint
    assert "grep_files" in hint


def test_format_hint_from_urls() -> None:
    hint = format_retrieval_batch_hint(
        "https://api.example.com/billing/report 有没有鉴权"
    )
    assert "billing" in hint
    assert "grep_files" in hint
