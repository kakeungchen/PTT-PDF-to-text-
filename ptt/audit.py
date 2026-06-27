"""抽样转换 PDF，并对 Markdown 输出做自动质检。

用法:
    python -m ptt.audit --pdf-dir pdf测试文档 --sample-size 4 --json

此入口面向回归抽检，不改变普通 CLI / GUI 行为。默认输出到 /tmp，
结束后清理临时转换结果；加 --keep-output 可保留现场便于人工复核。
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple

from .pipeline import convert


BAD_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("页眉水印残留", re.compile(
        r"(?m)保密资料|^(?!.*(?:编号|以下简称|制度|协议|内容)).*MTPS[-A-Z0-9]*.*$")),
    ("账号水印残留", re.compile(
        r"(?i)(?:bm[_\\-]*ch|chenj|chenjia|i?ang[o0]l|"
        r"1ang[o0]1|ang10|iaqlan9|陈加强|陈加\d?)")),
    ("低置信文字残留", re.compile(r"识别置信度低")),
    ("Markdown外部图片残留", re.compile(r"!\[[^\]]*\]\(<[^>]+>\)|_assets/|_assets\\\\")),
    ("未文本化图片占位残留", re.compile(r"图片区域未能可靠文本化|不写入外部图片")),
    ("申诉误识别", re.compile(r"中诉|甲诉|不子申诉")),
    ("为/沩误识别", re.compile(r"沩")),
    ("已/己误识别", re.compile(r"己离职|己经")),
    ("符号/数字误识别", re.compile(r"项//次|工有单天数|2 0:00|8 0%")),
    ("目录标点噪声", re.compile(r"[一二三四五六七八九十]{1,3}、[，,；;:：]")),
    ("目录孤立编号", re.compile(r"(?m)^[一二三四五六七八九十]{1,3}、\s*$")),
    ("孤立页码残留", re.compile(r"(?m)^\s*\d{1,3}\s*$")),
    ("计分语句误识别", re.compile(
        r"得分力|兜底力|(?:膨胀系数|系数|占比|比例|权重|天气等级|天气指数)力|"
        r"得分\s+-?\s*\d|签署准")),
    ("薪动力规则残留", re.compile(
        r"(?m)(?:^|\|)\s*0=\s*(?=\||$)|后白|留存日标|分数计算规\b|特场景|商服务费考核方案|"
        r"正常天气单恶劣天气单")),
    ("标题正文夹页码", re.compile(r"协议\s+\d{1,3}本协议内容")),
    ("编号段落断裂", re.compile(r"编号\s*\n\s*度”|主制\s*\n\s*度")),
    ("常见形近字残留", re.compile(
        r"提备|墻|因予|小金。太阳|虛假|末使用|酒漏|LOG0|1og0|"
        r"yan1i03|站点1习D|结算方窝|站点姐|强排抯|签暑|肯得牌|"
        r"亥月|KA導|考材|%\d{1,3}\b|站点\s*A[Iil]\b|A\d[.．]\s*A\d|"
	        r"肇月度|强排擅|站点组强推|集圴站|群合考核|则除异常单|汁算|方案）。|"
	        r"谢味\s*恕|特妹|特珠|特株|场意|算分不例|多倍权甫|权甫|"
	        r"300[,，zZ]|70046|级力\s*\d+|240天气|≤22|配送原因未完成率=Y2|XI|Xe|Yz|Zs|"
            r"张冢口|WKA品|品陳|品陈|"
	        r"结算方案[）)]|KA星级结算金|KA星级结\b|K[iI]\s*[、,，]\s*K[zZ][.．]\s*K[sS]|"
        r"K[iI]分\s*[、,，]\s*K[aA]分|K[.．]{2}K[aA]|K[Ii]分|"
        r"K[sS]分K[Nn]分|Q1\s*[.．]\s*Q2")),
    ("表格结构可读性问题", re.compile(
        r"呆站后|晋週场芳|别陈异吊早|超时时长=（2\s*\|\s*40|"
        r"虚假点送达率\s*得分Xs|Z5120|XT\s*\|\s*0|Z：\s*\|\s*120|"
        r"目标骑手达成天数/要求\s*考核周期得分该考核周期总天数|"
        r"介于门槛值、\s*介于\s*0%到100%之间\s*等比例计算得分目标值之间|"
        r"状态准")),
    ("乱码字符残留", re.compile(r"[�俁仴雲抇讳冏昇銷埃門哭伿怯奂沩亥導抯]")),
]

TABLE_FALLBACK_RE = re.compile(r"表格结构识别不稳定|!\[表格截图\]")


def list_pdfs(pdf_dir: str) -> List[str]:
    return sorted(
        os.path.join(pdf_dir, name)
        for name in os.listdir(pdf_dir)
        if name.lower().endswith(".pdf")
    )


def select_sample(pdfs: List[str], sample_size: int, offset: int) -> List[str]:
    if not pdfs:
        return []
    n = len(pdfs)
    size = min(sample_size, n)
    return [pdfs[(offset + i) % n] for i in range(size)]


def scan_markdown(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    issues = []
    for label, pattern in BAD_PATTERNS:
        matches = []
        for m in pattern.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            snippet = text[m.start():m.end()]
            matches.append({"line": line, "text": snippet[:80]})
            if len(matches) >= 8:
                break
        if matches:
            issues.append({"type": label, "matches": matches})
    _append_critical_missing_section_issues(text, issues)
    return {
        "path": path,
        "issue_count": sum(len(i["matches"]) for i in issues),
        "issues": issues,
        "table_fallbacks": len(TABLE_FALLBACK_RE.findall(text)),
    }


def _append_critical_missing_section_issues(text: str,
                                            issues: List[Dict[str, object]]) -> None:
    def add(label: str, needle: str):
        line = text.count("\n", 0, text.find(needle)) + 1 if needle in text else 1
        issues.append({
            "type": "关键内容缺失",
            "matches": [{"line": line, "text": label}],
        })

    compact = re.sub(r'\s+', '', text)
    if "5.4.7" in compact and "复合超时时长" in compact:
        has_formula = (
            r"\text{复合超时时长}" in text
            and r"A1_{\text{KA品牌单}}" in text
            and r"W_{\text{KA品牌单}}" in text
        )
        if not has_formula:
            add("5.4.7 复合超时时长缺少 A1/A2/A3/W 计算公式", "5.4.7")

    if "7.1" in compact and "KA品牌体验调分项" in compact:
        if "得分范围" not in text or "麦当劳完单量占比" not in text:
            add("7.1 KA品牌体验调分项缺少得分范围或指标定义", "7.1")

    if "8.1" in compact and "客诉虚假点送达" in compact:
        if "虚假点送达指" not in text or "客诉虚假点送达率" not in text:
            add("8.1 客诉虚假点送达缺少定义、数据来源或指标公式", "8.1")


def audit_pdf(pdf_path: str, out_dir: str) -> Dict[str, object]:
    progress_name = os.path.basename(pdf_path)

    def progress(msg, frac):
        print(f"\r[{frac*100:3.0f}%] {progress_name}: {msg}   ",
              end="", file=sys.stderr, flush=True)

    try:
        result = convert(pdf_path, out_dir, formats=("md",), progress=progress)
        print(file=sys.stderr)
    except Exception as e:
        print(file=sys.stderr)
        return {
            "source": pdf_path,
            "ok": False,
            "error": str(e),
            "markdown": None,
            "qa_issues": [],
            "warnings": [],
        }

    md_paths = [p for p in result.get("outputs", []) if p.endswith(".md")]
    md_scan = scan_markdown(md_paths[0]) if md_paths else {
        "path": None,
        "issue_count": 1,
        "issues": [{"type": "缺少 Markdown 输出", "matches": []}],
        "table_fallbacks": 0,
    }
    blocking_qa = [
        issue for issue in result.get("qa_issues", [])
        if "表格疑似列错位" not in issue
    ]
    ok = md_scan["issue_count"] == 0 and not blocking_qa
    return {
        "source": pdf_path,
        "ok": ok,
        "outputs": result.get("outputs", []),
        "pages": result.get("pages"),
        "blocks": result.get("blocks"),
        "flagged_blocks": result.get("flagged_blocks", 0),
        "warnings": result.get("warnings", []),
        "qa_issues": result.get("qa_issues", []),
        "blocking_qa": blocking_qa,
        "markdown": md_scan,
    }


def run_audit(pdf_dir: str, sample_size: int = 4, offset: int = 0,
              out_dir: str = None, keep_output: bool = False) -> Dict[str, object]:
    pdf_dir = os.path.abspath(pdf_dir)
    pdfs = list_pdfs(pdf_dir)
    sample = select_sample(pdfs, sample_size, offset)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or os.path.join(tempfile.gettempdir(), f"ptt_audit_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    results = [audit_pdf(path, out_dir) for path in sample]
    ok = bool(sample) and all(item["ok"] for item in results)
    report = {
        "ok": ok,
        "pdf_dir": pdf_dir,
        "total_pdfs": len(pdfs),
        "sample_size": len(sample),
        "offset": offset,
        "out_dir": out_dir if keep_output else None,
        "results": results,
    }
    if not keep_output:
        shutil.rmtree(out_dir, ignore_errors=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="ptt-audit",
        description="抽样转换 PDF 并自动检查 Markdown 质量")
    ap.add_argument("--pdf-dir", default="pdf测试文档", help="PDF 测试目录")
    ap.add_argument("--sample-size", type=int, default=4, help="本轮抽检数量")
    ap.add_argument("--offset", type=int, default=0, help="抽样起点")
    ap.add_argument("--out-dir", default=None, help="临时输出目录")
    ap.add_argument("--keep-output", action="store_true", help="保留临时输出")
    ap.add_argument("--json", action="store_true", help="JSON 输出")
    args = ap.parse_args(argv)

    report = run_audit(args.pdf_dir, args.sample_size, args.offset,
                       args.out_dir, args.keep_output)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        status = "通过" if report["ok"] else "发现问题"
        print(f"抽检{status}: {report['sample_size']}/{report['total_pdfs']}")
        for item in report["results"]:
            mark = "✓" if item["ok"] else "⚠"
            print(f"{mark} {os.path.basename(item['source'])}")
            md = item.get("markdown") or {}
            for issue in md.get("issues", []):
                print(f"  - {issue['type']}: {len(issue.get('matches', []))}处")
            for issue in item.get("blocking_qa", []):
                print(f"  - QA: {issue}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
