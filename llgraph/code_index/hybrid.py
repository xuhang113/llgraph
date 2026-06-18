"""兼容别名：search_hybrid 已更名为 search_parallel。"""

from llgraph.code_index.parallel_search import search_parallel

search_hybrid = search_parallel

__all__ = ["search_hybrid", "search_parallel"]
