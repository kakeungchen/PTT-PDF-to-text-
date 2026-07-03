"""命令行入口（也是智能体调用接口）。

用法:
    python -m ptt.cli 文件1.pdf [文件2.pdf ...] [-o 输出目录] [-f md docx] [--json]

--json 模式：进度打到 stderr，最终结果以 JSON 打到 stdout，方便 Agent 解析。
"""
import argparse
import json
import os
import sys

from . import __version__
from .pipeline import convert


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="ptt", description="PDF 转 Word/Markdown（本地 OCR，自动去水印/页眉页脚）")
    ap.add_argument("inputs", nargs="+", help="PDF 文件路径")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="输出目录（默认：与源文件同目录下的『转换结果』）")
    ap.add_argument("-f", "--formats", nargs="+", default=["md"],
                    choices=["md", "docx"], help="输出格式（默认只输出 Markdown）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果（Agent 模式）")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    results = []
    code = 0
    for path in args.inputs:
        if not os.path.isfile(path):
            print(f"找不到文件: {path}", file=sys.stderr)
            code = 1
            continue
        out_dir = args.out_dir or os.path.join(
            os.path.dirname(os.path.abspath(path)), "转换结果")

        def progress(msg, frac):
            print(f"\r[{frac*100:3.0f}%] {os.path.basename(path)}: {msg}   ",
                  end="", file=sys.stderr, flush=True)

        try:
            res = convert(path, out_dir, formats=tuple(args.formats),
                          progress=progress)
            print(file=sys.stderr)
            results.append(res)
            if not args.json:
                mark = "✓" if res.get("quality_ok", True) else "⚠"
                label = "转换完成" if res.get("quality_ok", True) else "转换完成，但质量审计需复核"
                print(f"{mark} {label}: {path}")
                for o in res["outputs"]:
                    print(f"   -> {o}")
                if res["flagged_blocks"]:
                    print(f"   ⚠ {res['flagged_blocks']} 处低置信内容已标注，建议人工核对")
                for issue in res.get("qa_issues", [])[:8]:
                    print(f"   ⚠ {issue}")
            if not res.get("quality_ok", True):
                code = 1
        except Exception as e:
            print(f"\n✗ {path}: {e}", file=sys.stderr)
            results.append({"source": path, "error": str(e)})
            code = 1
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
