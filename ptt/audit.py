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
    ("页内品牌标语残留", re.compile(r"(?:\d{0,3}\s*)?把世界[送医]到你手中")),
    ("低置信文字残留", re.compile(r"识别置信度低")),
    ("公式需核对", re.compile(r"公式原文（需核对）|需对照原 PDF 核对")),
    ("公式OCR残片", re.compile(
        r"[CPRTWY]KA品牌单指标释义|"
        r"虚假点送达率\s*[=＝]\s*TKA品牌单WKA品牌单|"
        r"配送原因未[完定]成率[^\n]*(?:Wex|PKA|KA[脚腳]M)|"
        r"A[L1I]?\s*KA品[牌解]单\s*\+\s*A2\s*KA品牌单\s*\+\s*A3\s*KA品[牌解]单")),
    ("旧式图片占位残留", re.compile(r"!\[(?:图片|表格截图|公式)\]\(<[^>]+>\)")),
    ("未文本化图片占位残留", re.compile(r"图片区域未能可靠文本化|不写入外部图片")),
    ("申诉误识别", re.compile(r"中诉|甲诉|不子申诉")),
    ("为/沩误识别", re.compile(r"沩")),
    ("已/己误识别", re.compile(r"己离职|己经")),
    ("符号/数字误识别", re.compile(
        r"项//次|工有单天数|2 0:00|8 0%|\\\|[〉＞]|KKA品牌体验总得分")),
    ("公式异常项目符号", re.compile(r"[；;][ \t]*[•·]")),
    ("目录标点噪声", re.compile(r"[一二三四五六七八九十]{1,3}、[，,；;:：]")),
    ("目录孤立编号", re.compile(r"(?m)^[一二三四五六七八九十]{1,3}、\s*$")),
    ("孤立页码残留", re.compile(r"(?m)^\s*\d{1,3}\s*$")),
    ("计分语句误识别", re.compile(
        r"得分力|兜底力|(?:膨胀系数|系数|占比|比例|权重|天气等级|天气指数)力|"
        r"得分\s+-?\s*\d|签署准|"
        r"(?:特殊|普通)场景体验得分\s*[-－—]\s*(?:特殊|普通)场景品牌订单")),
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
	        r"300[zZ]|70046|级力\s*\d+|240天气|≤22|配送原因未完成率=Y2|XI|Xe|Yz|Zs|"
            r"张冢口|品陳|品陈|美國配送|准到好用|门店门店流失|形象的行（|"
	        r"结算方案[）)]|KA星级结\b|K[iI]\s*[、,，]\s*K[zZ][.．]\s*K[sS]|"
        r"K[iI]分\s*[、,，]\s*K[aA]分|K[.．]{2}K[aA]|K[Ii]分|"
        r"K[sS]分K[Nn]分|Q1\s*[.．]\s*Q2")),
    ("表格结构可读性问题", re.compile(
        r"呆站后|晋週场芳|别陈异吊早|超时时长=（2\s*\|\s*40|"
        r"虚假点送达率\s*得分Xs|Z5120|XT\s*\|\s*0|Z：\s*\|\s*120|"
        r"目标骑手达成天数/要求\s*考核周期得分该考核周期总天数|"
        r"介于门槛值、\s*介于\s*0%到100%之间\s*等比例计算得分目标值之间|"
        r"状态准|商服务费并考核上算|惩资结约站|"
        r"违约责任不承担违场景|责任承担：站/月|"
        r"责任承担：上降低三档|美团配送滋时好用|"
        r"数据查计分规则询\s*ID|B1=0\s*7594|"
        r"(?<!最)近班级-新任城|(?:管理模块[；;：:].*){2}")),
    ("段落标签粘连", re.compile(
        r"违规处罚列表请注意|(?<!Q)94[：:]|剔除虚订单|"
        r"。计算\s*\n+\s*示例[:：]|考核“\s*\n+\s*品牌方口径准时率|"
        r"计算示例[:：]\s*\n+\s*示例[:：]|"
        r"数据来源[:：][^\n]{20,}数据播报[:：]|"
        r"数据播报[:：][^\n]{20,}降星规则[:：]|"
        r"出现以下行[，,]\s*影响|"
        r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*$|\d,\s+\d{3}\b", re.MULTILINE)),
    ("中国式表格错关联", re.compile(
        r"检核项目：(?:控|报|舍)(?:；|$)|"
        r"一级分类：站；检核项目：(?:建设|控)|"
        r"检核项目：建设；内容：.*标准\s*2[.．]\s*标准站|"
        r"^-\s*内容：[^\n]{8,}责任承担：|"
        r"(?:3[.．]\s*视频监|7[.．]\s*看板海|8[.．]\s*站内宿)\s+[\u4e00-\u9fff]|"
        r"遮挡\s*\d+\s*元/项/次[，,]\s*整改\s*4[.．]\s*流媒体|"
        r"员工宿[、，]\s*(?:整改)?舍|整改舍安全管理制度|"
        r"台账宿舍|安全\s*22[.．]\s*安全台账站外\s*23[.．]\s*选址安全|"
        r"一级分类：站点；检核项目：14[.．].*15[.．]\s*站点烟感|"
        r"一级分类：安全；检核项目：1[67][.．]|"
        r"内容：感消防联系人信息缺失|内容：设备（包含但不限于防火隔离|"
        r"(?<!整)改不达标需承担双|改需承担双|，整如|整例如|电，整器|"
        r"香改需承担双蕉|禁改不达标需承担双止|按《合作商安全管(?:；|$)|"
        r"内容：[^；\n]*?(?<!整改)不达标需承担双倍|"
        r"责任承担：\d+\s*元/项/次，?整改(?:违约金)?(?:。|$)",
        re.MULTILINE)),
    ("乱码字符残留", re.compile(r"[�俁仴雲抇讳冏昇銷埃門哭伿怯奂沩亥導抯]")),
]

TABLE_FALLBACK_RE = re.compile(r"表格结构识别不稳定|表格结构需核对|!\[表格截图\]")
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
        if label == "孤立页码残留":
            matches = _find_orphan_page_number_matches(text)
        else:
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
    _append_policy_table_context_issues(text, issues)
    return {
        "path": path,
        "issue_count": sum(len(i["matches"]) for i in issues),
        "issues": issues,
        "table_fallbacks": len(TABLE_FALLBACK_RE.findall(text)),
    }


def _find_orphan_page_number_matches(text: str) -> List[Dict[str, object]]:
    lines = text.splitlines()
    matches: List[Dict[str, object]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not re.fullmatch(r"\d{1,3}", stripped):
            continue
        if not _numeric_line_has_cjk_document_context(lines, idx):
            continue
        matches.append({"line": idx + 1, "text": stripped[:80]})
        if len(matches) >= 8:
            break
    return matches


def _numeric_line_has_cjk_document_context(lines: List[str], idx: int) -> bool:
    start = max(0, idx - 3)
    end = min(len(lines), idx + 4)
    context = "\n".join(lines[start:idx] + lines[idx + 1:end])
    if len(re.findall(r"[\u4e00-\u9fff]", context)) >= 4:
        return True
    return bool(re.search(r"目录|第\s*\d+\s*页|保密资料|协议|制度|考核", context))


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
            and r"R_{\text{KA品牌单}}" in text
        )
        if not has_formula:
            add("5.4.5 承托比缺少 R/W 计算公式", "5.4.5")

    if "5.4.6" in compact and "虚假点送达率" in compact:
        has_formula = (
            r"\text{虚假点送达率}" in text
            and r"T_{\text{KA品牌单}}" in text
        )
        has_context = all(item in compact for item in (
            "指标释义", "指标说明", "数据来源", "电话客诉", "风控抓取"
        ))
        if not has_formula:
            add("5.4.6 虚假点送达率缺少 T/W 计算公式", "5.4.6")
        if not has_context:
            add("5.4.6 虚假点送达率缺少指标释义、指标说明或数据来源", "5.4.6")

    if "7.1" in compact and "KA品牌体验调分项" in compact:
        if "得分范围" not in text or "麦当劳完单量占比" not in text:
            add("7.1 KA品牌体验调分项缺少得分范围或指标定义", "7.1")

    if "5.7" in compact and "特殊场景体验融合考核" in compact:
        required = [
            "考核方式",
            "考核指标",
            "天气等级",
            "考核规则",
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

    in_fenced_block = False
    for idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fenced_block = not in_fenced_block
            continue
        if in_fenced_block:
            continue
        if not stripped:
            continue
        if stripped.count("<br>") >= 2 or re.search(r"\|\s*[^|\n]*<br>", stripped):
            add("Markdown换行挤压", idx, stripped)
        if len(stripped) > 360 and not stripped.startswith("$$"):
            add("Markdown超长单行", idx, stripped)
        if re.search(r"站点组A麦当劳品\s*\|\s*牌特殊场景算分示例|融合后体\s*\|.*验得分",
                     stripped):
            add("表格标题断裂", idx, stripped)
        if (not stripped.startswith(("#", "|", "- "))
                and re.search(
                    r"[.!?。]\s+\d{1,2}(?:[.．]\d{1,2}){0,4}[.．]?\s+"
                    r"[A-Z][^\n]{2,90}$",
                    stripped)):
            add("标题粘连段落", idx, stripped)
        if (not stripped.startswith(("#", "|", "$$", "```"))
                and re.search(
                    r"[𝑎-𝑧𝐴-𝑍α-ωΑ-Ω](?:[a-z]|[A-Z][a-z])|"
                    r"\)[𝑎-𝑧𝐴-𝑍α-ωΑ-Ω](?:[a-z]|[A-Z][a-z])",
                    stripped)):
            add("数学变量与正文粘连", idx, stripped)

    section_71 = _extract_section(text, "7.1")
    if section_71:
        for idx, line in section_71:
            if "<br>" in line or len(line.strip()) > 360:
                add("7.1 排版挤压", idx, line.strip())
                break


def _append_policy_table_context_issues(text: str,
                                        issues: List[Dict[str, object]]) -> None:
    def add(line_no: int, snippet: str) -> None:
        for issue in issues:
            if issue["type"] == "中国式表格上下文异常":
                issue["matches"].append({"line": line_no, "text": snippet[:80]})
                return
        issues.append({
            "type": "中国式表格上下文异常",
            "matches": [{"line": line_no, "text": snippet[:80]}],
        })

    for line_no, block_lines in _iter_policy_blocks(text):
        block = "\n".join(block_lines)
        first = block_lines[0].strip()
        fields = _policy_block_fields(block_lines)
        category = fields.get("一级分类", "")
        item = fields.get("检核项目", "")
        content = fields.get("内容", "")
        responsibility = fields.get("责任承担", "")

        if "；检核项目：" in first or "；内容：" in first:
            add(line_no, first)
        if category in {"内容", "项目", "检核项目", "责任承担", "说明"}:
            add(line_no, block)
        if not item:
            add(line_no, block)
        if item in {"控", "报", "舍", "建设", "配置"}:
            add(line_no, block)
        if re.search(r'^(?:感|设备|屏幕|理规范|违约金)', content):
            add(line_no, block)
        if re.search(r'\d{1,2}[.．]\s*[\u4e00-\u9fff]{1,8}\s+\S', content):
            add(line_no, block)
        if responsibility and _policy_responsibility_tail_bad(responsibility):
            add(line_no, block)


def _iter_policy_blocks(text: str) -> List[Tuple[int, List[str]]]:
    lines = text.splitlines()
    blocks: List[Tuple[int, List[str]]] = []
    current: List[str] = []
    start_line = 0
    for idx, line in enumerate(lines, start=1):
        if line.startswith("- 一级分类："):
            if current:
                blocks.append((start_line, current))
            current = [line]
            start_line = idx
            continue
        if current and (line.startswith("  ") or not line.strip()):
            if line.strip():
                current.append(line)
            continue
        if current:
            blocks.append((start_line, current))
            current = []
            start_line = 0
    if current:
        blocks.append((start_line, current))
    return blocks


def _policy_block_fields(block_lines: List[str]) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    current = ""
    labels = ("一级分类", "检核项目", "内容", "责任承担")
    for raw in block_lines:
        line = raw.strip()
        if line.startswith("- "):
            line = line[2:].strip()
        matched = False
        for label in labels:
            prefix = f"{label}："
            if line.startswith(prefix):
                fields[label] = line[len(prefix):].strip()
                current = label
                matched = True
                break
        if matched:
            continue
        if current:
            fields[current] = (fields.get(current, "") + " " + line).strip()
    return fields


def _policy_responsibility_tail_bad(text: str) -> bool:
    compact = re.sub(r'\s+', '', text or "").rstrip("。；;，,")
    return bool(
        compact.endswith(("整", "整改", "整改违约金"))
        or re.search(r'元/项/次，?整改(?:违约金)?$', compact)
    )


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
    blocking_qa = list(result.get("blocking_qa") or result.get("qa_issues", []))
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
        "qa_warnings": result.get("qa_warnings", []),
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

    def case_ka_star_settlement_amount_allowed() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("商服务费 = 基础服务费 + KA星级结算金额 + KA体验膨胀费\n")
            result = scan_markdown(path)
        _assert_equal(result["issue_count"], 0, "issue_count")

    def case_formula_review_marker_blocks_pass() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("**公式原文（需核对）**\n\n客诉虚假点送达率=分子/分母\n")
            result = scan_markdown(path)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_true("公式需核对" in issue_types, "公式需核对")

    def case_formula_ocr_fragment_blocks_pass() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("指标释义：配送人员虚假点送达率= TKA品牌单WKA品牌单\n")
            result = scan_markdown(path)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_true("公式OCR残片" in issue_types, "公式OCR残片")

    def case_formula_bullet_glue_blocks_pass() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("肯德基完单量占比；•肯德基体验得分\n")
            result = scan_markdown(path)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_true("公式异常项目符号" in issue_types, "公式异常项目符号")

    def case_normal_semicolon_then_next_bullet_allowed() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("•电话客诉：可通过线上申诉；\n•风控抓取：可通过线上申诉。\n")
            result = scan_markdown(path)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_false("公式异常项目符号" in issue_types, "公式异常项目符号")

    def case_trailing_heading_glued_to_paragraph_blocks_pass() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "This paragraph should end cleanly, but it has a glued "
                    "section title at the end. 5.4. Long-horizon Parsing\n"
                )
            result = scan_markdown(path)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_true("标题粘连段落" in issue_types, "标题粘连段落")

    def case_inline_math_word_glue_blocks_pass() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("Each token attends to the preceding 𝑛output tokens.\n")
            result = scan_markdown(path)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_true("数学变量与正文粘连" in issue_types, "数学变量与正文粘连")

    def case_convert_outputs_single_md_without_assets_or_layout() -> None:
        import fitz
        from .pipeline import convert

        with tempfile.TemporaryDirectory() as td:
            pdf_path = os.path.join(td, "sample.pdf")
            out_dir = os.path.join(td, "out")
            doc = fitz.open()
            page = doc.new_page(width=300, height=200)
            page.insert_text((40, 80), "单文件 Markdown 输出检查")
            doc.save(pdf_path)
            doc.close()

            result = convert(pdf_path, out_dir, formats=("md",))
            names = os.listdir(out_dir)

        _assert_equal(len(result["outputs"]), 1, "outputs")
        _assert_true(result["outputs"][0].endswith(".md"), "md output")
        _assert_false(any("_assets" in name for name in names), "assets output")
        _assert_false(any("_layout" in name for name in names), "layout output")

    def case_pipeline_table_warning_blocks_after_audit() -> None:
        import fitz
        from .pipeline import convert
        from . import qa as qa_module

        original_scan = qa_module.qa_scan

        def fake_scan(blocks):
            return ["块1 表格疑似列错位，建议人工复核"], 1

        qa_module.qa_scan = fake_scan
        try:
            with tempfile.TemporaryDirectory() as td:
                pdf_path = os.path.join(td, "sample.pdf")
                out_dir = os.path.join(td, "out")
                doc = fitz.open()
                page = doc.new_page(width=300, height=200)
                page.insert_text((40, 80), "表格风险提示分类检查")
                doc.save(pdf_path)
                doc.close()

                result = convert(pdf_path, out_dir, formats=("md",))
            _assert_false(result["quality_ok"], "quality_ok")
            _assert_true(result["qa_issues"], "qa_issues")
            _assert_true(result["qa_warnings"], "qa_warnings")
        finally:
            qa_module.qa_scan = original_scan

    def case_resolved_fallback_warning_does_not_block() -> None:
        from .models import Block
        from .pipeline import _blocking_issues_from_warnings

        warning = "低置信区域已转为截图(y=120): 承托比 RKA品牌单"
        block = Block(
            kind="image",
            text="承托比=RKA品牌单/WKA品牌单",
            bbox=(0, 100, 200, 160),
            flags=["auto_image", "formula"],
        )
        markdown = "$$\\text{承托比}=\\frac{R_{\\text{KA品牌单}}}{W_{\\text{KA品牌单}}}$$"
        issues = _blocking_issues_from_warnings(
            [warning], blocks=[block], markdown_text=markdown)
        _assert_false(issues, "resolved fallback warning")

    def case_unresolved_fallback_warning_still_blocks() -> None:
        from .models import Block
        from .pipeline import _blocking_issues_from_warnings

        warning = "低置信区域已转为截图(y=120): 无法识别的重要公式"
        block = Block(
            kind="image", text="无法识别的重要公式",
            bbox=(0, 100, 200, 160), flags=["auto_image", "formula"],
        )
        issues = _blocking_issues_from_warnings(
            [warning], blocks=[block], markdown_text="# 输出\n")
        _assert_true(issues, "unresolved fallback warning")

    def case_discarded_toc_fallback_warning_does_not_block() -> None:
        from .models import Block
        from .pipeline import _blocking_issues_from_warnings

        warning = (
            "低置信区域已转为截图(y=120): "
            "二、管理框架⋯⋯1三、违约责任说明⋯⋯8"
        )
        block = Block(
            kind="image", text="", bbox=(0, 100, 300, 180),
            flags=["auto_image", "discarded_toc_fragment"],
        )
        issues = _blocking_issues_from_warnings(
            [warning], blocks=[block], markdown_text="# 正文\n")
        _assert_false(issues, "discarded toc fallback warning")

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
                    "考核方式：普通场景和特殊场景分开考核。\n"
                    "考核指标：虚假点送达率、配送原因未完成率、复合准时率。\n"
                    "天气等级：10、20、30、40。\n"
                    "考核规则：按运单首次调度天气等级判定。\n"
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

    def case_english_chart_ticks_not_orphan_page_numbers() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "Figure 3 shows cache hit ratio across model sizes.\n"
                    "18\n16\n14\n12\n10\n"
                    "Duration and throughput stay stable after warmup.\n"
                )
            result = scan_markdown(path)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_false("孤立页码残留" in issue_types, "孤立页码残留")

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

    def case_disinfection_table_spill_blocks_pass() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "责任承担：违约责任不承担违场景 1 >=80%约责任。\n"
                    "责任承担：站/月\n"
                    "责任承担：上降低三档承担违约责任。\n"
                    "内容：美团配送滋时好用考核周期内无提交记录。\n"
                )
            result = scan_markdown(path)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_true("表格结构可读性问题" in issue_types,
                     "表格结构可读性问题")

    def case_chinese_policy_table_misalignment_regression() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "- 一级分类：站；检核项目：控；内容：合作商未按要求配送站点购买视频监控且未报备，导致美；3.视频监 团检核人员无法及时查看站内情况。\n"
                    "- 内容：流媒体设备出现遮挡 200元/项/次，整改 4.流媒体 屏幕。\n"
                    "- 内容：合作商配送站点公告栏中展示的配送服务人员健康证存在虚假。；责任承担：需按照《合作商用工管理规范》相关约定承担违约责任。\n"
                )
            result = scan_markdown(path)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_true("中国式表格错关联" in issue_types, "中国式表格错关联")

    def case_policy_table_context_regression() -> None:
        with tempfile.TemporaryDirectory() as td:
            bad = os.path.join(td, "bad.md")
            with open(bad, "w", encoding="utf-8") as f:
                f.write(
                    "- 一级分类：健康证；检核项目：1.健康证；内容：健康证存在虚假。；责任承担：需承担违约责任\n"
                    "- 一级分类：内容\n"
                    "  检核项目：特殊说明：\n"
                    "  内容：说明被塞进分类列。\n"
                    "- 一级分类：标准站\n"
                    "  检核项目：控\n"
                    "  内容：合作商未报备。\n"
                )
            result = scan_markdown(bad)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_true("中国式表格上下文异常" in issue_types, "中国式表格上下文异常")

    def case_policy_table_context_valid() -> None:
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, "good.md")
            with open(good, "w", encoding="utf-8") as f:
                f.write(
                    "- 一级分类：健康证\n"
                    "  检核项目：1.健康证\n"
                    "  内容：合作商配送站点公告栏中展示的配送服务人员健康证存在虚假。\n"
                    "  责任承担：需按照《合作商用工管理规范》相关约定承担违约责任\n"
                    "- 一级分类：标准站\n"
                    "  检核项目：3.视频监控\n"
                    "  内容：合作商未按要求购买视频监控。\n"
                    "  责任承担：200元/项/次，整改不达标需承担双倍违约金。\n"
                )
            result = scan_markdown(good)
        issue_types = [issue["type"] for issue in result["issues"]]
        _assert_false("中国式表格上下文异常" in issue_types, "中国式表格上下文异常")

    return [
        ("audit.select_sample_wraps", case_select_sample_wraps),
        ("audit.markdown_patterns", case_markdown_patterns),
        ("audit.salary_policy_residuals", case_salary_policy_residuals),
        ("audit.normal_formula_not_zero_reversal", case_normal_formula_not_zero_reversal),
        ("audit.normal_y2_variable_allowed", case_normal_y2_variable_allowed),
        ("audit.ka_star_settlement_amount_allowed", case_ka_star_settlement_amount_allowed),
        ("audit.formula_review_marker_blocks_pass", case_formula_review_marker_blocks_pass),
        ("audit.formula_ocr_fragment_blocks_pass", case_formula_ocr_fragment_blocks_pass),
        ("audit.formula_bullet_glue_blocks_pass", case_formula_bullet_glue_blocks_pass),
        ("audit.normal_semicolon_then_next_bullet_allowed", case_normal_semicolon_then_next_bullet_allowed),
        ("audit.trailing_heading_glued_to_paragraph_blocks_pass", case_trailing_heading_glued_to_paragraph_blocks_pass),
        ("audit.inline_math_word_glue_blocks_pass", case_inline_math_word_glue_blocks_pass),
        ("audit.convert_outputs_single_md_without_assets_or_layout", case_convert_outputs_single_md_without_assets_or_layout),
        ("audit.pipeline_table_warning_blocks_after_audit", case_pipeline_table_warning_blocks_after_audit),
        ("audit.resolved_fallback_warning_does_not_block", case_resolved_fallback_warning_does_not_block),
        ("audit.unresolved_fallback_warning_still_blocks", case_unresolved_fallback_warning_still_blocks),
        ("audit.discarded_toc_fallback_warning_does_not_block", case_discarded_toc_fallback_warning_does_not_block),
        ("audit.critical_sections_missing", case_critical_sections_missing),
        ("audit.critical_sections_complete", case_critical_sections_complete),
        ("audit.table_readability_regression", case_table_readability_regression),
        ("audit.disinfection_table_spill_blocks_pass", case_disinfection_table_spill_blocks_pass),
        ("audit.chinese_policy_table_misalignment_regression", case_chinese_policy_table_misalignment_regression),
        ("audit.policy_table_context_regression", case_policy_table_context_regression),
        ("audit.policy_table_context_valid", case_policy_table_context_valid),
        ("audit.markdown_readability_regression", case_markdown_readability_regression),
        ("audit.english_chart_ticks_not_orphan_page_numbers", case_english_chart_ticks_not_orphan_page_numbers),
    ]


def run_builtin_checks(stream=None) -> Dict[str, object]:
    from . import assemble, coverage, export, normalize, qa, text_extract

    def add_module_cases(module) -> None:
        getter = getattr(module, "builtin_check_cases", None)
        if getter:
            cases.extend(getter())

    cases: List[Tuple[str, Callable[[], None]]] = []
    cases.extend(_audit_builtin_cases())
    for module in (normalize, qa, assemble, export, coverage, text_extract):
        add_module_cases(module)

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
    elif not ok:
        report["out_dir"] = out_dir
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
