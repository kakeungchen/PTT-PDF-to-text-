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
from typing import Callable, Dict, List, Optional, Tuple


BAD_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("页眉水印残留", re.compile(
        r"(?m)保密资料|^(?!.*(?:编号|以下简称|制度|协议|内容)).*MTPS[-A-Z0-9]*.*$")),
    ("账号水印残留", re.compile(
        r"(?i)(?:bm[_\\-]*ch|chenj|chenjia|i?ang[o0]l|"
        r"1ang[o0]1|ang10|iaqlan9|陈加强|陈加\d?)")),
    ("低置信文字残留", re.compile(r"识别置信度低")),
    ("旧式图片占位残留", re.compile(r"!\[(?:图片|表格截图|公式)\]\(<[^>]+>\)")),
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
            r"张冢口|品陳|品陈|"
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
SECTION_HEADING_RE = re.compile(r"^#{1,6}\s+")


def _run_case(name: str, fn: Callable[[], None]) -> Optional[Dict[str, str]]:
    try:
        fn()
        return None
    except Exception as exc:  # pragma: no cover - 汇总错误信息用
        return {
            "test": name,
            "message": str(exc) or exc.__class__.__name__,
        }


def _assert_equal(actual, expected, label: str = "") -> None:
    if actual != expected:
        prefix = f"{label}: " if label else ""
        raise AssertionError(f"{prefix}expected {expected!r}, got {actual!r}")


def _assert_true(value, label: str = "") -> None:
    if not value:
        prefix = f"{label}: " if label else ""
        raise AssertionError(f"{prefix}expected truthy value, got {value!r}")


def _assert_false(value, label: str = "") -> None:
    if value:
        prefix = f"{label}: " if label else ""
        raise AssertionError(f"{prefix}expected falsy value, got {value!r}")


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
    _append_markdown_readability_issues(text, issues)
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

    if "5.4.5" in compact and "承托比" in compact:
        has_formula = (
            r"\text{承托比}" in text
            and (r"R_{\text{KA品牌单}}" in text or "![公式原图]" in text)
        )
        if not has_formula:
            add("5.4.5 承托比缺少 R/W 计算公式或公式原图", "5.4.5")

    if "5.4.6" in compact and "虚假点送达率" in compact:
        has_formula = (
            r"\text{虚假点送达率}" in text
            and (r"T_{\text{KA品牌单}}" in text or "![公式原图]" in text)
        )
        has_context = all(item in compact for item in (
            "指标释义", "指标说明", "数据来源", "电话客诉", "风控抓取"
        ))
        if not has_formula:
            add("5.4.6 虚假点送达率缺少 T/W 计算公式或公式原图", "5.4.6")
        if not has_context:
            add("5.4.6 虚假点送达率缺少指标释义、指标说明或数据来源", "5.4.6")

    if "7.1" in compact and "KA品牌体验调分项" in compact:
        if "得分范围" not in text or "麦当劳完单量占比" not in text:
            add("7.1 KA品牌体验调分项缺少得分范围或指标定义", "7.1")

    if "5.7" in compact and "特殊场景体验融合考核" in compact:
        required = [
            "考核目标",
            "考核目标举例",
            "普通场景体验满分目标",
            "特殊场景体验满分目标",
            "核算公式",
            "特殊场景完成单占比",
            "普通场景算分示例",
            "特殊场景算分示例",
            "融合后体验得分",
            "124.50",
            "107.80",
            "122.9818",
        ]
        missing = [item for item in required if item not in compact]
        if missing:
            add("5.7 特殊场景体验融合考核缺少：" + "、".join(missing), "5.7")

    if "8.1" in compact and "客诉虚假点送达" in compact:
        if "虚假点送达指" not in text or "客诉虚假点送达率" not in text:
            add("8.1 客诉虚假点送达缺少定义、数据来源或指标公式", "8.1")


def _append_markdown_readability_issues(text: str,
                                        issues: List[Dict[str, object]]) -> None:
    def add(issue_type: str, line_no: int, snippet: str) -> None:
        for issue in issues:
            if issue["type"] == issue_type:
                issue["matches"].append({"line": line_no, "text": snippet[:80]})
                return
        issues.append({
            "type": issue_type,
            "matches": [{"line": line_no, "text": snippet[:80]}],
        })

    for idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.count("<br>") >= 2 or re.search(r"\|\s*[^|\n]*<br>", stripped):
            add("Markdown换行挤压", idx, stripped)
        if len(stripped) > 360 and not stripped.startswith("$$"):
            add("Markdown超长单行", idx, stripped)
        if re.search(r"站点组A麦当劳品\s*\|\s*牌特殊场景算分示例|融合后体\s*\|.*验得分",
                     stripped):
            add("表格标题断裂", idx, stripped)

    section_71 = _extract_section(text, "7.1")
    if section_71:
        for idx, line in section_71:
            if "<br>" in line or len(line.strip()) > 360:
                add("7.1 排版挤压", idx, line.strip())
                break


def _extract_section(text: str, heading_token: str) -> List[Tuple[int, str]]:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if SECTION_HEADING_RE.match(line) and heading_token in re.sub(r'\s+', '', line):
            start = idx
            break
    if start is None:
        return []
    out: List[Tuple[int, str]] = []
    for idx in range(start, len(lines)):
        if idx > start and SECTION_HEADING_RE.match(lines[idx]):
            break
        out.append((idx + 1, lines[idx]))
    return out


def audit_pdf(pdf_path: str, out_dir: str) -> Dict[str, object]:
    from .pipeline import convert

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


def _audit_builtin_cases() -> List[Tuple[str, Callable[[], None]]]:
    def case_select_sample_wraps() -> None:
        pdfs = ["a.pdf", "b.pdf", "c.pdf"]
        _assert_equal(
            select_sample(pdfs, sample_size=4, offset=2),
            ["c.pdf", "a.pdf", "b.pdf"],
        )

    def case_markdown_patterns() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "正常文本\n"
                    "1\n"
                    "这里有 MTPS-ZS-LY-V66 和 中诉\n"
                    "四、； 服务费结算方案\n"
                    "得分力0分\n"
                    "![图片](<out_assets/fig1.png>)\n"
                    "<!-- 图片区域未能可靠文本化，已按要求不写入外部图片 -->\n"
                    "展示沩大网单，结算方窝，膨胀系数力0.8\n"
                    "补充协议 2本协议内容\n"
                    "<!-- 表格结构识别不稳定，已保留原表格截图，建议以截图为准 -->\n"
                    "![表格截图](<a.png>)\n"
                )
            result = scan_markdown(path)
        _assert_true(result["issue_count"] >= 2, "issue_count")
        _assert_equal(result["table_fallbacks"], 2, "table_fallbacks")
        _assert_equal(
            [issue["type"] for issue in result["issues"]],
            [
                "页眉水印残留",
                "旧式图片占位残留",
                "未文本化图片占位残留",
                "申诉误识别",
                "为/沩误识别",
                "目录标点噪声",
                "孤立页码残留",
                "计分语句误识别",
                "标题正文夹页码",
                "常见形近字残留",
                "乱码字符残留",
            ],
        )

    def case_salary_policy_residuals() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "| 不考核 | ≤8分钟 | 0= |\n"
                    "| 总完成单（W） | 站点组剔除异常单后白 | 总完成单 |\n"
                    "| 负向反馈率 | 天气等级力10 | 正常天气单恶劣天气单 |\n"
                    "留存日标达成率，分数计算规 分数区间，普通场景和特场景融合，"
                    "烽火台-商服务费考核方案\n"
                )
            result = scan_markdown(path)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_true("计分语句误识别" in issue_types, "计分语句误识别")
        _assert_true("薪动力规则残留" in issue_types, "薪动力规则残留")

    def case_normal_formula_not_zero_reversal() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("$$\\text{达成率}=100=100\\%$$\n")
            result = scan_markdown(path)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_false("薪动力规则残留" in issue_types, "薪动力规则残留")

    def case_normal_y2_variable_allowed() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("有效在线时长≥Y2小时，标准2服务质量达标。\n")
            result = scan_markdown(path)
        _assert_equal(result["issue_count"], 0, "issue_count")

    def case_critical_sections_missing() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "#### 5.4.7 复合超时时长\n"
                    "| 订单类型 | 复合超时时长订单口径 |\n"
                    "|---|---|\n"
                    "#### 5.7 特殊场景体验融合考核的说明\n"
                    "特殊场景算分示例\n"
                    "#### 7.1 KA 品牌体验调分项\n"
                    "计分规则1：\n"
                    "#### 8.1 站点组【客诉虚假点送达】降星\n"
                    "降星规则：\n"
                )
            result = scan_markdown(path)
        critical = [i for i in result["issues"] if i["type"] == "关键内容缺失"]
        _assert_equal(len(critical), 4, "关键内容缺失")

    def case_critical_sections_complete() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "#### 5.4.7 复合超时时长\n"
                    "$$\\text{复合超时时长}=\\frac{A1_{\\text{KA品牌单}}+A2_{\\text{KA品牌单}}+A3_{\\text{KA品牌单}}}{W_{\\text{KA品牌单}}}$$\n"
                    "#### 5.7 特殊场景体验融合考核的说明\n"
                    "考核目标举例：不同场景的体验目标。\n"
                    "普通场景体验满分目标，特殊场景体验满分目标。\n"
                    "核算公式：融合后体验得分按照特殊场景完成单占比计算。\n"
                    "普通场景算分示例，普通场景得分124.50。\n"
                    "特殊场景算分示例，特殊场景得分107.80，融合后体验得分122.9818。\n"
                    "#### 7.1 KA 品牌体验调分项\n"
                    "得分范围：[0,0.2]\n"
                    "麦当劳完单量占比=麦当劳完单量/总量\n"
                    "#### 8.1 站点组【客诉虚假点送达】降星\n"
                    "虚假点送达指骑手未将餐品/货品按照订单要求送达指定位置虚假点击送达的行为。\n"
                    "客诉虚假点送达率=命中客诉虚假点送达单量/完成单量\n"
                )
            result = scan_markdown(path)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_false("关键内容缺失" in issue_types, "关键内容缺失")

    def case_markdown_readability_regression() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "#### 7.1 KA 品牌体验调分项\n"
                    "| 指标定义1 | 第一行<br>第二行<br>第三行 |\n"
                    "|  | 站点组A麦当劳品 | 牌特殊场景算分示例 |\n"
                    + "很长" * 220 + "\n"
                )
            result = scan_markdown(path)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_true("Markdown换行挤压" in issue_types, "Markdown换行挤压")
        _assert_true("Markdown超长单行" in issue_types, "Markdown超长单行")
        _assert_true("表格标题断裂" in issue_types, "表格标题断裂")
        _assert_true("7.1 排版挤压" in issue_types, "7.1 排版挤压")

    def case_table_readability_regression() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "| 例：呆站后 | A，在晋週场芳 | 定考核日早 |\n"
                    "|  | 复合 | 超时时长=（2 | 40+1,320+4， |\n"
                    "虚假点送达率 得分Xs 0 Ys 115 Z5120\n"
                    "| XT | 0 |\n"
                    "| Z： | 120 |\n"
                    "【KA 品牌驻点骑手】目标骑手达成天数/要求 考核周期得分该考核周期总天数\n"
                    "介于门槛值、 介于 0%到100%之间 等比例计算得分目标值之间\n"
                    "是否属于同配送区同商情况以考核周期最后一天状态准。\n"
                )
            result = scan_markdown(path)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_true("表格结构可读性问题" in issue_types, "表格结构可读性问题")

    return [
        ("audit.select_sample_wraps", case_select_sample_wraps),
        ("audit.markdown_patterns", case_markdown_patterns),
        ("audit.salary_policy_residuals", case_salary_policy_residuals),
        ("audit.normal_formula_not_zero_reversal", case_normal_formula_not_zero_reversal),
        ("audit.normal_y2_variable_allowed", case_normal_y2_variable_allowed),
        ("audit.critical_sections_missing", case_critical_sections_missing),
        ("audit.critical_sections_complete", case_critical_sections_complete),
        ("audit.table_readability_regression", case_table_readability_regression),
        ("audit.markdown_readability_regression", case_markdown_readability_regression),
    ]


def run_builtin_checks(stream=None) -> Dict[str, object]:
    from . import assemble, coverage, export, normalize, qa

    cases: List[Tuple[str, Callable[[], None]]] = []
    cases.extend(_audit_builtin_cases())
    cases.extend(normalize.builtin_check_cases())
    cases.extend(qa.builtin_check_cases())
    cases.extend(assemble.builtin_check_cases())
    cases.extend(export.builtin_check_cases())
    cases.extend(coverage.builtin_check_cases())

    failures: List[Dict[str, str]] = []
    for name, fn in cases:
        failure = _run_case(name, fn)
        if failure:
            failures.append(failure)

    result = {
        "ok": not failures,
        "tests_run": len(cases),
        "failures": failures,
    }
    if stream is not None:
        mark = "通过" if result["ok"] else "失败"
        print(f"内建回归检查{mark}: {result['tests_run']}项", file=stream)
        for failure in failures[:10]:
            print(f"  - {failure['test']}: {failure['message']}", file=stream)
    return result


def run_audit(pdf_dir: str, sample_size: int = 4, offset: int = 0,
              out_dir: str = None, keep_output: bool = False,
              skip_selfcheck: bool = False) -> Dict[str, object]:
    pdf_dir = os.path.abspath(pdf_dir)
    pdfs = list_pdfs(pdf_dir)
    sample = select_sample(pdfs, sample_size, offset)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or os.path.join(tempfile.gettempdir(), f"ptt_audit_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    if skip_selfcheck:
        selfcheck = {
            "ok": True,
            "skipped": True,
            "tests_run": 0,
            "failures": [],
        }
    else:
        selfcheck = run_builtin_checks(stream=sys.stderr)

    results = [audit_pdf(path, out_dir) for path in sample]
    ok = selfcheck["ok"] and bool(sample) and all(item["ok"] for item in results)
    report = {
        "ok": ok,
        "pdf_dir": pdf_dir,
        "total_pdfs": len(pdfs),
        "sample_size": len(sample),
        "offset": offset,
        "out_dir": out_dir if keep_output else None,
        "selfcheck": selfcheck,
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
    ap.add_argument("--skip-selfcheck", action="store_true",
                    help="跳过内建回归自查，只跑 PDF 抽检")
    ap.add_argument("--json", action="store_true", help="JSON 输出")
    args = ap.parse_args(argv)

    report = run_audit(args.pdf_dir, args.sample_size, args.offset,
                       args.out_dir, args.keep_output, args.skip_selfcheck)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        status = "通过" if report["ok"] else "发现问题"
        print(f"抽检{status}: {report['sample_size']}/{report['total_pdfs']}")
        selfcheck = report.get("selfcheck", {})
        if selfcheck.get("skipped"):
            print("- 内建回归检查: 已跳过")
        else:
            mark = "✓" if selfcheck.get("ok") else "⚠"
            print(f"{mark} 内建回归检查: {selfcheck.get('tests_run', 0)}项")
            for failure in selfcheck.get("failures", [])[:5]:
                print(f"  - {failure['test']}: {failure['message']}")
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
