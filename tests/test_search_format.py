"""检索结果 Cursor 风格格式化。"""

from llgraph.code_index.search_format import format_search_hit, truncate_search_snippet


def test_truncate_search_snippet_short():
    assert truncate_search_snippet("public void foo()") == "public void foo()"


def test_truncate_search_snippet_long():
    text = "x" * 100
    out = truncate_search_snippet(text)
    assert len(out) == 80
    assert out.endswith("…")


def test_truncate_search_snippet_multiline():
    assert truncate_search_snippet("line1\nline2") == "line1 line2"


def test_format_search_hit_single_line():
    line = format_search_hit(1, "repo/Foo.java:42", "public void syncOrg()")
    assert line == "1. repo/Foo.java:42  public void syncOrg()"
    assert "\n" not in line


def test_format_search_hit_no_snippet():
    assert format_search_hit(2, "bar.py:10", "") == "2. bar.py:10"
