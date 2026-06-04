"""llgraph index 子命令。"""

import sys
from pathlib import Path

from llgraph.code_index.index_dispatch import dispatch_index


def main(argv: list[str] | None = None) -> None:
    """
    索引 CLI 入口。

    @param argv 参数列表（不含程序名）
    """
    workspace = Path(".").resolve()
    outcome = dispatch_index(workspace, argv, bare_means_status=False)
    sys.exit(outcome.exit_code)


if __name__ == "__main__":
    main(sys.argv[1:])
