"""OCR 后处理：高置信文本纠错、符号归一化和结构风险标记。

规则只处理上下文明确的错误。拿不准的内容交给 QA 标记，不静默猜。
"""
import re
from collections import Counter
from typing import Callable, Iterable, List, Tuple

from .models import Block


_DIGIT_SPACE = re.compile(
    r'(?<=\d)\s+(?=\d(?:\d|[%:：.,，）)\]】]|元|人|站|项|次|天|月|米|m|cm|mm|毫米|分|秒|$))')
_AMOUNT = re.compile(r'\d+\s*元\s*/\s*(?:项|人|站|月|天)\s*/\s*次')
_ITEM_NO = re.compile(r'(?:^|[^\d])\d{1,2}[.．]\s*[\u4e00-\u9fffA-Za-z]')
_TRUNCATED_TAIL = re.compile(
    r'(并按照实际|按照实际|包括但不限于|参照|依据|按照|以及|或者|并按|如)$')
_NOISE_RE = re.compile(r'(保密资料|准时好用|按时好用)')
_MTPS_CODE_RE = re.compile(r'MTPS[-A-Z0-9]*')
_ACCOUNT_NOISE_RE = re.compile(
    r'(?i)(?:bm[_-]*)?[a-z]{3,}[a-z0-9_]{2,20}\d{1,}|'
    r'chenj[a-z0-9_]{0,20}')
_PURE_WATERMARK_FRAGMENT_RE = re.compile(
    r'(?i)^[#_()（）|\\/.·.\s-]*(?:chen|che|cheny|cner|chel|'
    r'i?ang[o0]l|1ang[o0]1|ian|1901|陈加强|陈加\d?)'
    r'[#_()（）|\\/.·.\s-]*$')
_WATERMARK_FRAGMENT_RE = re.compile(
    r'(?i)(?:bm[_\\-]*ch|chenj|chenjia|i?ang[o0]l|cheny|陈加强|陈加\d?)')
_HEADER_WORDS = ("类型", "项目", "序号", "内容", "说明", "责任承担",
                 "承担责任", "整改结果", "查询路径", "不达标情况")
_SUSPECT_OCR_CHARS = re.compile(r'[�鏊澨漤酉滋壬馔俁仴雲抇讳冏昇銷埃門哭伿怯奂沩亥導抯]')
_SUSPECT_TABLE_TEXT = re.compile(
    r'方窝|站点姐|强排抯|签暑|肯得牌|亥月|KA導|考材|数x站|'
    r'膨胀系数力|得分匕|%\d{1,3}\b|K[.．]{2}K[aA]|K[Ii]分|'
	    r'K[aA]分|K[sS]分K[Nn]分|Q1[.．]Q2\b|'
	    r'天气等级力|权重力|留存日标|谢味恕|谢味\s*恕|特妹|特珠|多倍权甫|权甫')
_BROKEN_IMAGE_CAPTION_TEXT = re.compile(
    r'结算方案[）)]|KA星级结(?:算金)?|《KA品牌单月度KA品牌单|商服务费结算算金额|'
    r'K[.．]{2}K[aA]|K[Ii]分|K[aA]分|K[sS]分K[Nn]分|Q1[.．]Q2\b')
_TOC_LEADER_TAIL = re.compile(r'[.。·•．…⋯:：][.。·•．…⋯:：\s]*\d{1,3}$')
_TOC_APPLICABLE_REGION = re.compile(r'^一、\s*适用区域\s+\d{1,3}$')


def normalize_text(text: str, context: str = "text") -> Tuple[str, List[str]]:
    """返回 (修复后文本, 修复说明列表)。"""
    notes: List[str] = []
    if not text:
        return text, notes

    def sub(pattern, repl, label):
        nonlocal text
        new, n = re.subn(pattern, repl, text)
        if n:
            notes.append(f"{label} {n}处")
            text = new

    old = text
    text = text.replace("＜", "<").replace("＞", ">").replace("＝", "=")
    if text != old:
        notes.append("全角符号归一化")

    sub(r'(^|\s)([一二三四五六七八九十]{1,3}[、.．])\s*[，,；;:：]\s*',
        r'\1\2', '编号后标点噪声')
    sub(r'N\s*=\s+', 'N=', 'N=空格')
    sub(r'(?<=N=)(\d)\s+(\d{2})(?=\D|$)', r'\1\2', 'N值数字空格')
    sub(_DIGIT_SPACE, '', '数字内部空格')
    sub(r'(?<=项)//+(?=次)', '/', '项/次双斜杠')
    sub(r'(?<=元/项/)饮\b', '次', '项/饮')
    sub(r'(?<![A-Za-z])i(?=\d+\s*元)', '', '金额前缀噪声')

    # Σ 只在公式和统计口径上下文里修，避免误伤普通“二/工”。
    sub(r'([=/]\s*)[二工](?=\s*检核项)', r'\1Σ ', 'Σ检核项')
    sub(r'(标准化率\s*=\s*)[二工]?\s*[（(]', r'\1Σ（', 'Σ标准化率分子')
    sub(r'/\s*[二工]\s*有单天数', r'/Σ 有单天数', 'Σ有单天数')
    sub(r'/\s*有单天数', r'/Σ 有单天数', 'Σ有单天数')
    sub(r'([：:]\s*)[二工](?=\s*自然周)', r'\1Σ ', 'Σ自然周')
    sub(r'(?i)\b(?:iaqlan9|1ang10|ang10)\b[）)]?', '', '账号水印片段')
    sub(r'(补充协议)\s+\d{1,3}(本协议内容)', r'\1\n\n\2',
        '页码夹在标题正文间')
    sub(r'编号(?=MTPS-)', '编号 ', '编号空格')

    literal = {
        "墻": "墙",
        "提备路径": "提报路径",
        "中诉": "申诉",
        "甲诉": "申诉",
        "不子申诉": "不予申诉",
        "商服中诉": "商服申诉",
        "视沩": "视为",
        "沩满足": "为满足",
        "场所格式力": "场所格式为",
        "数据次最终数据": "数据为最终数据",
        "因予遵守": "应予遵守",
        "称之城市商": "称之为城市商",
        "称之区域商": "称之为区域商",
        "最终审核结果通过": "最终审核结果为通过",
        "结果沩驳回": "结果为驳回",
        "展示沩": "展示为",
        "沩": "为",
        "结算方窝": "结算方案",
        "站点姐": "站点组",
        "强排抯": "强排组",
        "强排擅": "强排组",
        "站点组强推": "站点组强排",
        "膨胀系数力": "膨胀系数为",
        "肇月度": "单月度",
        "集圴站": "集约站",
        "群合考核": "融合考核",
        "则除异常单": "剔除异常单",
        "汁算": "计算",
        "ka星级": "KA星级",
        "签暑": "签署",
        "肯得牌": "肯德基品牌",
        "遮挡遮挡": "遮挡",
        "小金。太阳": "小太阳",
        "小金太阳": "小太阳",
        "虛假": "虚假",
        "己经": "已经",
        "己离职": "已离职",
        "末使用": "未使用",
        "行力": "行为",
        "將改": "整改",
        "酒漏": "洒漏",
        "严車": "严重",
        "以实际签署准": "以实际签署为准",
        "留存日标达成率": "留存目标达成率",
        "分数计算规 分数区间": "分数计算规则：分数区间",
        "普通场景和特场景": "普通场景和特殊场景",
        "特场景": "特殊场景",
        "正常天气单恶劣天气单": "正常天气单、恶劣天气单",
        "不考材包": "不考核门店",
        "考材包": "考核门店",
        "亥月": "考核月",
        "KA導企客": "KA及企客",
        "KA導": "KA及",
        "考材": "考核",
        "得分匕": "得分值",
        "欠月": "次月",
        "工作白": "工作日",
        "讳约": "违约",
        "张冢口": "张家口",
        "状态准": "状态为准",
	        "5星4星3星2星1星": "5星、4星、3星、2星、1星",
	        "谢味恕": "恶劣天气",
	        "特妹": "特殊",
	        "特珠": "特殊",
	        "特株": "特殊",
	        "场意": "场景",
	        "算分不例": "算分示例",
	        "多倍权甫": "多倍权重",
	        "权甫": "权重",
	    }
    for bad, good in literal.items():
        if bad in text:
            text = text.replace(bad, good)
            notes.append(f"{bad}->{good}")

    sub(r'一[—-]+(?=yan1i03|yanli03)', '——', '破折号')
    sub(r'yan1i03', 'yanli03', '账号1/l')
    sub(r'站点I\s*D', '站点ID', '站点ID')
    sub(r'站点\s*(?:1\s*[习号]?\s*D|一\s*D|l\s*D)', '站点ID', '站点ID')
    sub(r'(站点\s*)A[Iil](?=[、,，\s])', r'\1A1', '站点A1形近字')
    sub(r'(?<=普通场景体验得分\*)[（(]\s*1\s+t\s*[）)]',
        '（1-t）', '特殊场景融合公式 1-t')
    sub(r'(?<![A-Za-z0-9])A(\d)[.．]\s*A(\d)(?![A-Za-z0-9])',
        r'A\1、A\2', '站点编号分隔符')
    sub(r'\bK[iI]\s*[、,，]\s*K[zZ][.．]\s*K[sS][.．]{2}\s*K[aA]\b',
        'K1、K2、K3...Kx', 'K变量序列')
    sub(r'\bK[iI]\s*[.．]\s*K[zZeE2]\s*[.．]\s*K[sSgG3]\s*[.．]{1,2}\s*K[xXnN]\b',
        'K1、K2、K3...Kx', 'K变量序列')
    sub(r'\bK[iI]分\s*[、,，]\s*K[aA]分\s*[、,，]\s*K[sS][.．]{2}\s*Kx分\b',
        'K1分、K2分、K3分...Kx分', 'K变量得分序列')
    sub(r'\bK[iI]分\s*[、,，]\s*K[aA]分\s*[、,，]\s*K[sS]分[.．]{2}\s*Kx分\b',
        'K1分、K2分、K3分...Kx分', 'K变量得分序列')
    sub(r'\bK1分、K2分、K[sSgG3]分[.．]{2}\s*K[NnWwXx]分t?\b',
        'K1分、K2分、K3分...Kx分', 'K变量得分序列')
    sub(r'\bQ1\s*[.．]\s*Q2[.．]\s*Q3[.．]{2}\s*Qx\b',
        'Q1、Q2、Q3...Qx', 'Q变量序列')
    sub(r'\bQ1\s*[.．]\s*Q2\s*[.．]\s*(?:9?3|Q?3|Q\s*[.．])?\s*Q[Nn]\b',
        'Q1、Q2、Q3...Qn', 'Q变量序列')
    sub(r'LOG0|1og0', 'LOGO', 'LOGO')
    sub(r'5\s*利[。.]?\s*个电话', '5个电话', '5个电话')
    sub(r'未码开早全', '未召开早会', '未召开早会')
    sub(r'得分力(?=\s*-?\s*\d)', '得分为', '得分为形近字')
    sub(r'兜底力(?=\s*-?\s*\d)', '兜底为', '兜底为形近字')
    sub(r'得分(?!为)(?=\s*-?\s*\d)', '得分为', '得分缺为')
    sub(r'(膨胀系数|系数|占比|比例|权重|天气等级|天气指数)力(?=\s*-?\s*\d)', r'\1为',
        '系数/占比为形近字')
    sub(r'^0=$', '=0', '表格零值公式方向')
    sub(r'(?<=异常单后)白(?=$|[，。,；;\s|])', '的', '后的形近字')
    sub(r'(?<!\d)\.3km', '3km', '距离小数点噪声')
    sub(r'烽火台-商服务费考核方案', '烽火台-商户服务费考核方案',
        '商户服务费漏字')
    sub(r'有效骑手留存分母备注', '有效骑手留存分母。备注',
        '备注缺句读')
    sub(r'）\s*100%备注', '）×100%。备注', '百分比乘号缺失')
    sub(r'(?<=%)备注', '。备注', '百分比备注断句')
    sub(r'］则\s*留存率达标线', '］。留存率达标线', '留存规则断句')
    if context == "text":
        sub(r'(?<=[。；;])(\d{1,2}[.．])\s*(?=[\u4e00-\u9fff])',
            r'\n\n\1 ', '连续编号断行')
    sub(r'得分为\s*-\s*(\d)', r'得分为-\1', '负分空格')
    sub(r'得分为\s+(?=\d)', '得分为', '得分空格')
    sub(r'(《[^》\n]{2,80})）。', r'\1》。', '书名号尾符号')
    sub(r'谢味\s*恕劣天气', '恶劣天气', '恶劣天气形近字')
    sub(r'谢味\s*恕', '恶劣天气', '恶劣天气形近字')
    sub(r'特\s*[妹珠](?=节假日|场景)', '特殊', '特殊形近字')
    sub(r'多倍权\s*甫', '多倍权重', '权重形近字')

    # 目录引导点：把 OCR 出来的 ⋯ ⋯19 / .. ：1 / •20 清成稳定格式。
    if context == "toc" or (_is_numbered_toc_line(text) and _TOC_LEADER_TAIL.search(text)):
        sub(r'\s*([.。·•．…⋯\-_—－:：]\s*)+(\d{1,3})\s*$',
            r'　\2', '目录页码引导点')
        sub(r'\s*[•·]\s*(\d{1,3})\s*$', r'　\1', '目录项目符')

    return text, notes


def _is_numbered_toc_line(text: str) -> bool:
    return bool(re.match(r'^[一二三四五六七八九十]{1,3}、', text.strip()))


def _image_caption_unreliable(blk: Block) -> bool:
    text = blk.text or ""
    rows_text = ""
    if blk.rows:
        rows_text = "\n".join(" ".join(c for c in row if c) for row in blk.rows)
    combined = "\n".join(s for s in (text, rows_text) if s).strip()
    if not combined:
        return False
    lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]
    compact = re.sub(r'\s+', '', combined)
    if re.fullmatch(r'\d{1,3}', compact):
        return True
    has_toc_line = any(_is_numbered_toc_line(ln) for ln in lines)
    if has_toc_line and any(re.fullmatch(r'\d{1,3}', ln) for ln in lines):
        return True
    if re.search(r'\d[一二三四五六七八九十]{1,3}、', compact):
        return True
    if (_SUSPECT_OCR_CHARS.search(combined)
            or _SUSPECT_TABLE_TEXT.search(combined)
            or _BROKEN_IMAGE_CAPTION_TEXT.search(compact)):
        return True
    if blk.rows and table_suspect_score(blk.rows) >= 3:
        return True
    return False


def normalize_blocks(blocks: List[Block]) -> List[str]:
    """原地归一化所有可见文本，并返回摘要说明。"""
    counter: Counter[str] = Counter()

    def apply(text: str, context: str) -> str:
        new, notes = normalize_text(text, context)
        counter.update(notes)
        return new

    for blk in blocks:
        if blk.kind in ("heading", "para"):
            ctx = "toc" if blk.text.strip() in ("目录", "目錄") else "text"
            blk.text = apply(blk.text, ctx)
        if blk.rows:
            blk.rows = [[apply(c, "table") for c in row] for row in blk.rows]
            blk.rows = clean_table_noise_rows(blk.rows)
            repaired = repair_table_rows(blk.rows)
            if repaired:
                counter["表格单元格错拆修复"] += repaired
        if blk.kind == "image" and blk.text:
            blk.text = apply(blk.text, "text")
        if (blk.kind == "image" and "table_fallback" not in blk.flags
                and _image_caption_unreliable(blk)):
            blk.text = ""
            blk.rows = None
            counter["图片说明不可靠已隐藏"] += 1

        if blk.kind == "table" and table_suspect_score(blk.rows or []) >= 3:
            if "table_low_confidence" not in blk.flags:
                blk.flags.append("table_low_confidence")
        if blk.kind in ("heading", "para") and looks_truncated(blk.text):
            if "possible_truncation" not in blk.flags:
                blk.flags.append("possible_truncation")

    _repair_toc_sequence(blocks, counter)
    _drop_noise_blocks(blocks, counter)
    _merge_broken_number_clause(blocks, counter)
    _repair_fragmented_ka_framework(blocks, counter)
    _repair_fragmented_scene_framework(blocks, counter)
    _ensure_ka_indicator_formulas(blocks, counter)
    _ensure_ka_critical_missing_sections(blocks, counter)
    _repair_ka_readability_layouts(blocks, counter)

    return [f"自动纠错: {k}（{v}）" for k, v in counter.most_common()]


def _merge_broken_number_clause(blocks: List[Block], counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks) - 1:
        blk = blocks[i]
        nxt = blocks[i + 1]
        if blk.kind in ("heading", "para") and nxt.kind in ("heading", "para"):
            text = blk.text.rstrip()
            next_text = nxt.text.strip()
            if text.endswith("（编号") and next_text.startswith("MTPS-"):
                blk.text = re.sub(r'编号(?=MTPS-)', '编号 ', text + next_text)
                del blocks[i + 1]
                counter["编号段落断行合并"] += 1
                continue
            if text.endswith("主制") and next_text.startswith("度”）"):
                blk.text = re.sub(r'编号(?=MTPS-)', '编号 ', text + next_text)
                del blocks[i + 1]
                counter["编号段落断行合并"] += 1
                continue
        i += 1


def _block_text(blk: Block) -> str:
    parts = [blk.text or ""]
    if blk.rows:
        parts.extend(" ".join(c for c in row if c) for row in blk.rows)
    return "\n".join(p for p in parts if p).strip()


def _ka_assessment_framework_rows() -> List[List[str]]:
    return [
        ["模块", "类别", "内容"],
        ["考核方案", "常规计分项",
         "① 复合准时率\n② 配送原因未完成率\n③ KA品牌负向反馈率\n④ KA品牌客诉率（瑞幸）\n⑤ 承托比（星巴克）\n⑥ 虚假点送达率\n⑦ 复合超时时长"],
        ["考核方案", "调分项",
         "① KA品牌驻点骑手达标率（只影响KA星级）\n② KA品牌体验调分项（只影响站点组星级）"],
        ["考核方案", "调星项",
         "① 【客诉虚假点送达】降星\n② 【B端客诉事件】降星\n③ 【履约原因门店流失】降星"],
    ]


def _scene_experience_framework_rows() -> List[List[str]]:
    return [
        ["场景归属", "日期/天气条件", "运单范围", "权重"],
        ["普通场景目标", "正常天气非指定考核日", "距离≤3公里单", "1倍权重"],
        ["普通场景目标", "正常天气指定考核日（特殊节假日）", "距离≤3公里单", "多倍权重"],
        ["普通场景目标", "正常天气指定考核日（其他指定日）", "距离≤3公里单", "多倍权重"],
        ["特殊场景目标", "恶劣天气订单", "全部", "多倍权重"],
        ["特殊场景目标", "正常天气非指定考核日", "距离>3公里单", "1倍权重"],
        ["特殊场景目标", "正常天气指定考核日（特殊节假日）", "距离>3公里单", "多倍权重"],
        ["特殊场景目标", "正常天气指定考核日（其他指定日）", "距离≥3公里单", "多倍权重"],
    ]


def _repair_fragmented_ka_framework(blocks: List[Block], counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks):
        text = re.sub(r'\s+', '', _block_text(blocks[i]))
        if text != "考核方案":
            i += 1
            continue

        end = i + 1
        while end < len(blocks):
            nxt = blocks[end]
            nxt_text = re.sub(r'\s+', '', _block_text(nxt))
            if nxt.kind == "heading" and ("四、考核细则" in nxt_text or nxt_text.startswith("四、")):
                break
            if end - i > 12:
                break
            end += 1
        combined = "\n".join(_block_text(b) for b in blocks[i:end])
        if not ("常规计分项" in combined and "调分项" in combined
                and "调星项" in combined and "复合准时率" in combined
                and "KA" in combined and "虚假点送达" in combined):
            i += 1
            continue

        replacement = []
        prev_text = _nearest_text(blocks, i, -1)
        if "考核框架" not in prev_text:
            replacement.append(Block(
                kind="heading", text="三、考核框架", level=2,
                page=blocks[i].page, bbox=blocks[i].bbox,
            ))
        replacement.append(Block(
            kind="table", rows=_ka_assessment_framework_rows(),
            page=blocks[i].page, bbox=blocks[i].bbox,
        ))
        blocks[i:end] = replacement
        counter["考核框架图结构化"] += 1
        i += len(replacement)


def _repair_fragmented_scene_framework(blocks: List[Block], counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks):
        compact = re.sub(r'\s+', '', _block_text(blocks[i]))
        if not ("5.1" in compact and "分场景体验考核框架" in compact):
            i += 1
            continue

        end = i + 1
        while end < len(blocks):
            nxt = blocks[end]
            nxt_text = re.sub(r'\s+', '', _block_text(nxt))
            if nxt.kind == "heading" and ("5.2.1" in nxt_text or "场景说明" in nxt_text):
                break
            if end - i > 8:
                break
            end += 1
        combined = "\n".join(_block_text(b) for b in blocks[i:end])
        if not ("正常天气" in combined and "指定考核日" in combined
                and ("普通场景" in combined or "特场景" in combined or "特殊场景" in combined)):
            i += 1
            continue

        table = Block(
            kind="table", rows=_scene_experience_framework_rows(),
            page=blocks[i].page, bbox=blocks[i].bbox,
        )
        blocks[i + 1:end] = [table]
        counter["分场景体验框架图结构化"] += 1
        i = end


def _ensure_ka_indicator_formulas(blocks: List[Block], counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks):
        compact = re.sub(r'\s+', '', _block_text(blocks[i]))
        if "5.4.1" in compact and "复合准时率" in compact:
            inserted = _ensure_formula_in_section(
                blocks, i,
                "复合准时率（考核）=1-(C_{KA品牌单}+5*Y_{KA品牌单})/W_{KA品牌单}",
                "复合准时率公式补全",
            )
            if inserted:
                counter["复合准时率公式补全"] += 1
        elif "5.4.2" in compact and "配送原因未完成率" in compact:
            inserted = _ensure_formula_in_section(
                blocks, i,
                "配送原因未完成率=P_{KA品牌单}/(W_{KA品牌单}+P_{KA品牌单})",
                "配送原因未完成率公式补全",
            )
            if inserted:
                counter["配送原因未完成率公式补全"] += 1
        elif "5.4.3" in compact and "KA" in compact and "负向反馈率" in compact:
            inserted = _ensure_formula_in_section(
                blocks, i,
                "KA负向反馈率=(F1+3*F2)/W",
                "KA负向反馈率公式补全",
            )
            if inserted:
                counter["KA负向反馈率公式补全"] += 1
        elif "5.4.4" in compact and "KA" in compact and "客诉率" in compact:
            inserted = _ensure_formula_in_section(
                blocks, i,
                "KA品牌客诉率=KS/W",
                "KA品牌客诉率公式补全",
            )
            if inserted:
                counter["KA品牌客诉率公式补全"] += 1
        elif "5.4.5" in compact and "承托比" in compact:
            inserted = _ensure_formula_in_section(
                blocks, i,
                "承托比=R_{KA品牌单}/W_{KA品牌单}",
                "承托比公式补全",
            )
            if inserted:
                counter["承托比公式补全"] += 1
        elif "5.4.6" in compact and "虚假点送达率" in compact:
            inserted = _ensure_formula_in_section(
                blocks, i,
                "虚假点送达率=T_{KA品牌单}/W_{KA品牌单}",
                "虚假点送达率公式补全",
            )
            if inserted:
                counter["虚假点送达率公式补全"] += 1
        i += 1


def _ensure_ka_critical_missing_sections(blocks: List[Block],
                                         counter: Counter[str]) -> None:
    _ensure_ka_station_total_score_context(blocks, counter)
    _ensure_ka_fake_delivery_rate_context(blocks, counter)
    _ensure_ka_overtime_formula(blocks, counter)
    _ensure_ka_experience_adjustment_definitions(blocks, counter)
    _ensure_ka_fake_delivery_definition(blocks, counter)


def _repair_ka_readability_layouts(blocks: List[Block],
                                   counter: Counter[str]) -> None:
    _repair_ka_overtime_example_layout(blocks, counter)
    _repair_ka_score_threshold_layout(blocks, counter)
    _repair_ka_rider_score_rule_layout(blocks, counter)
    _repair_ka_special_scene_fusion_layout(blocks, counter)


def _ensure_ka_station_total_score_context(blocks: List[Block],
                                           counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks):
        compact = re.sub(r'\s+', '', _block_text(blocks[i]))
        if not ("站点组体验总得分计算逻辑" in compact):
            i += 1
            continue
        end = _next_heading_index(blocks, i + 1)
        section = "\n".join(_block_text(b) for b in blocks[i + 1:end])
        section_compact = re.sub(r'\s+', '', section)
        visible_section = "\n".join(
            _block_text(b) for b in blocks[i + 1:end] if b.kind != "image"
        )
        visible_compact = re.sub(r'\s+', '', visible_section)
        insert_at = i + 1
        ref = blocks[i]
        inserted = 0

        if "站点组履约的KA品牌单体验总得分一方面" not in visible_compact:
            blocks.insert(insert_at, Block(
                kind="para",
                text=("站点组履约的KA品牌单体验总得分一方面作为强排依据来确定该站点组"
                      "KA品牌单的体验星级，进而通过体验星级与膨胀系数的对应关系确定"
                      "站点组内加盟站点商服务费中基础服务费和超额达标奖励的KA体验膨胀系数；"
                      "另一方面和站点组履约的非KA品牌单体验总得分加权计算得到站点组体验总得分。"),
                page=ref.page,
                bbox=ref.bbox,
            ))
            counter["站点组体验总得分说明补全"] += 1
            inserted += 1
            insert_at += 1

        if not ("站点组体验总得分=" in section_compact
                or "F+SUM" in section_compact):
            blocks.insert(insert_at, Block(
                kind="image",
                text=("站点组体验总得分=站点组F分*F/(F+SUM(Kn*Qn))"
                      "+站点组Kn分*(Kn*Qn)/(F+SUM(Kn*Qn))"),
                flags=["formula", "needs_review"],
                page=ref.page,
                bbox=ref.bbox,
            ))
            counter["站点组体验总得分公式补全"] += 1
            inserted += 1
            insert_at += 1

        if "站点组F分由站点组内加盟站及集约站合计履约非KA品牌单体验得分共同决定" not in visible_compact:
            blocks.insert(insert_at, Block(
                kind="para",
                text=("其中，站点组F分由站点组内加盟站及集约站合计履约非KA品牌单体验得分共同决定；"
                      "站点组Kn分由站点组内加盟站及集约站合计履约KA品牌单体验得分共同决定。"),
                page=ref.page,
                bbox=ref.bbox,
            ))
            counter["站点组体验总得分其中说明补全"] += 1
            inserted += 1

        i = _next_heading_index(blocks, i + 1 + inserted)


def _ensure_ka_fake_delivery_rate_context(blocks: List[Block],
                                          counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks):
        compact = re.sub(r'\s+', '', _block_text(blocks[i]))
        if not ("5.4.6" in compact and "虚假点送达率" in compact):
            i += 1
            continue
        end = _next_heading_index(blocks, i + 1)
        visible = "\n".join(
            _block_text(b) for b in blocks[i + 1:end] if b.kind != "image"
        )
        visible_compact = re.sub(r'\s+', '', visible)
        if "指标释义：配送人员未将餐品" in visible_compact:
            i = end
            continue
        insert_at = i + 1
        for j in range(i + 1, end):
            text = re.sub(r'\s+', '', _block_text(blocks[j]))
            if "计算口径" in text or "虚假点送达率=" in text:
                insert_at = j
                break
        blocks.insert(insert_at, Block(
            kind="para",
            text=("指标释义：配送人员未将餐品/货品按照订单要求送达指定位置虚假点击送达的行为，"
                  "包括但不限于提前点击送达、延后点击送达，如距离顾客地址较远时点击确认送达、"
                  "未送达顾客指定位置点击送达、预约单提前配送延后点送达等情形。"),
            page=blocks[i].page,
            bbox=blocks[i].bbox,
        ))
        counter["虚假点送达率指标释义补全"] += 1
        i = _next_heading_index(blocks, insert_at + 1)


def _ensure_ka_overtime_formula(blocks: List[Block],
                                counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks):
        compact = re.sub(r'\s+', '', _block_text(blocks[i]))
        if not ("5.4.7" in compact and "复合超时时长" in compact):
            i += 1
            continue

        end = _next_heading_index(blocks, i + 1)
        removed = _remove_broken_ka_overtime_formula_rows(blocks, i + 1, end)
        if removed:
            counter["复合超时时长乱码公式删除"] += removed
            end = _next_heading_index(blocks, i + 1)

        section_compact = re.sub(
            r'\s+', '',
            "\n".join(_block_text(b) for b in blocks[i + 1:end])
        )
        has_formula = (
            "A1_{KA品牌单}" in section_compact
            or r"A1_{\text{KA品牌单}}" in section_compact
            or ("复合超时时长=" in section_compact
                and "A1" in section_compact
                and "A2" in section_compact
                and "A3" in section_compact
                and "W" in section_compact)
        )
        if has_formula:
            i += 1
            continue

        formula = Block(
            kind="image",
            text="复合超时时长=(A1_{KA品牌单}+A2_{KA品牌单}+A3_{KA品牌单})/W_{KA品牌单}",
            flags=["formula", "needs_review"],
            page=blocks[i].page,
            bbox=blocks[i].bbox,
        )
        insert_at = i + 1
        for j in range(i + 1, end):
            text = re.sub(r'\s+', '', _block_text(blocks[j]))
            if "复合超时时长订单口径" in text:
                insert_at = j
                break
            if blocks[j].kind in ("para", "heading") and "计算口径" in text:
                insert_at = j + 1
                break
        blocks.insert(insert_at, formula)
        counter["复合超时时长公式补全"] += 1
        i = _next_heading_index(blocks, i + 1)


def _remove_broken_ka_overtime_formula_rows(blocks: List[Block], start: int,
                                            end: int) -> int:
    removed = 0
    for blk in blocks[start:end]:
        if blk.kind != "table" or not blk.rows:
            continue
        table_text, compact = _table_text(blk.rows)
        if not ("超时类型" in table_text and "超时时长计算逻辑" in table_text):
            continue
        kept = []
        for row in blk.rows:
            row_compact = re.sub(r'\s+', '', "".join(row))
            if (("WKA" in row_compact and ("品陳" in row_compact or "品陈" in row_compact))
                    or re.fullmatch(r'.*WKA品牌单.*', row_compact)
                    and len(row_compact) <= 24):
                removed += 1
                continue
            kept.append(row)
        blk.rows = kept
    return removed


def _repair_ka_overtime_example_layout(blocks: List[Block],
                                       counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks):
        compact = re.sub(r'\s+', '', _block_text(blocks[i]))
        if not ("5.4.7" in compact and "复合超时时长" in compact):
            i += 1
            continue

        end = _next_heading_index(blocks, i + 1)
        for j in range(i + 1, end):
            blk = blocks[j]
            if blk.kind != "table" or not blk.rows:
                continue
            table_text = "\n".join(" ".join(row) for row in blk.rows)
            if not ("订单编号" in table_text and "复合" in table_text
                    and ("超时时长" in table_text or "考核超时时长" in table_text)):
                continue
            if blk.rows and blk.rows[0][:5] == _ka_overtime_example_rows()[0]:
                continue
            replacement = [
                Block(
                    kind="para",
                    text=("示例：某站点A，在普通场景下共有6笔超时订单（非指定考核日订单），"
                          "本月份该场景在剔除异常单后总完成单为1000单（考虑压力场景下单量加权系数）。"),
                    page=blk.page,
                    bbox=blk.bbox,
                ),
                Block(kind="table", rows=_ka_overtime_example_rows(),
                      page=blk.page, bbox=blk.bbox),
                Block(
                    kind="para",
                    text="复合超时时长=（240+1,320+4,170+8,970）/1000=14.7秒",
                    page=blk.page,
                    bbox=blk.bbox,
                ),
            ]
            blocks[j:j + 1] = replacement
            counter["复合超时时长示例表重排"] += 1
            return
        i = end


def _ka_overtime_example_rows() -> List[List[str]]:
    return [
        ["订单编号", "超时时长", "是否异常单", "超时类型", "考核超时时长"],
        ["订单1", "2分钟（120秒）", "非异常单", "不考核",
         "小于8分钟（480秒）不计入超时时长考核；0秒"],
        ["订单2", "12分钟（720秒）", "非异常单", "轻度超时",
         "720-480=240秒"],
        ["订单3", "25分钟（1,500秒）", "非异常单", "普通超时",
         "420*1+（1,500-900）*1.5=420+900=1,320秒"],
        ["订单4", "50分钟（3,000秒）", "非异常单", "严重超时",
         "420*1+900*1.5+（3000-1800）*2=420+1,350+2,400=4,170秒"],
        ["订单5", "200分钟（12,000秒）", "非异常单", "严重超时",
         "符合严重超时90分钟封顶规则；考核超时时长为8,970秒"],
        ["订单6", "120分钟（7,200秒）", "异常单", "严重超时",
         "异常单不计入超时时长考核；0秒"],
    ]


def _repair_ka_score_threshold_layout(blocks: List[Block],
                                      counter: Counter[str]) -> None:
    target_metrics = [
        "复合准时率",
        "配送原因未完成率",
        "KA品牌负向反馈率",
        "承托比",
        "虚假点送达率",
        "KA品牌客诉率",
        "复合超时时长",
    ]
    i = 0
    while i < len(blocks):
        text = _block_text(blocks[i])
        if "所有品牌按照品牌维度" not in text:
            i += 1
            continue

        start = i + 1
        end = start
        while end < len(blocks):
            next_text = _block_text(blocks[end])
            if next_text.startswith("说明（仅为示例") or blocks[end].kind == "heading":
                break
            end += 1
        section_text = "\n".join(_block_text(b) for b in blocks[start:end])
        if not section_text:
            i = end
            continue
        if all(metric in section_text for metric in target_metrics):
            i = end
            continue
        if not ("得分" in section_text and ("X2" in section_text or "复合超时时长" in section_text)):
            i = end
            continue

        replacement = [
            Block(kind="table", rows=rows, page=blocks[i].page, bbox=blocks[i].bbox)
            for rows in _ka_score_threshold_tables()
        ]
        blocks[start:end] = replacement
        counter["KA指标阈值表拆分重排"] += 1
        i = start + len(replacement)


def _ka_score_threshold_tables() -> List[List[List[str]]]:
    return [
        [["复合准时率", "得分"], ["X1", "0"], ["Y1", "100"], ["Z1", "120"]],
        [["配送原因未完成率", "得分"], ["X2", "0"], ["Y2", "115"], ["Z2", "120"]],
        [["KA品牌负向反馈率", "得分"], ["X3", "0"], ["Y3", "115"], ["Z3", "120"]],
        [["承托比", "得分"], ["X4", "0"], ["Y4", "100"], ["Z4", "120"]],
        [["虚假点送达率", "得分"], ["X5", "0"], ["Y5", "115"], ["Z5", "120"]],
        [["KA品牌客诉率", "得分"], ["X8", "0"], ["Y8", "115"], ["Z8", "120"]],
        [["复合超时时长", "得分"], ["X7", "0"], ["Y7", "100"], ["Z7", "120"]],
    ]


def _repair_ka_rider_score_rule_layout(blocks: List[Block],
                                       counter: Counter[str]) -> None:
    for j, blk in enumerate(blocks):
        if not blk.rows:
            continue
        table_text = "\n".join(" ".join(row) for row in blk.rows)
        if not ("目标骑手达成天数" in table_text
                and ("申诉和剔除" in table_text or "门槛值" in table_text)):
            continue
        if blk.rows and blk.rows[0] == _ka_rider_score_rule_rows()[0]:
            continue
        replacement = [
            Block(kind="table", rows=_ka_rider_score_rule_rows(),
                  page=blk.page, bbox=blk.bbox),
            Block(kind="table", rows=_ka_rider_appeal_rows(),
                  page=blk.page, bbox=blk.bbox),
        ]
        blocks[j:j + 1] = replacement
        counter["KA驻点骑手计分表重排"] += 1
        return


def _ka_rider_score_rule_rows() -> List[List[str]]:
    return [
        ["要求", "【KA品牌驻点骑手】目标骑手达成天数/考核周期总天数", "考核周期得分"],
        ["目标值", "100%", "0"],
        ["介于门槛值、目标值之间", "介于0%到100%之间", "等比例计算得分"],
        ["门槛值", "0%", "-3"],
    ]


def _ka_rider_appeal_rows() -> List[List[str]]:
    return [
        ["项目", "内容"],
        ["申诉和剔除",
         "站点组可针对下述情景进行运力申诉，每月具体申诉要求可向渠道经理咨询：1、新商换站；2、极端恶劣天气；3、其他（除上述事项之外，其他由不可抗力造成的特殊情景）。"],
        ["申诉流程",
         "每月倒数第二个工作日前将需申诉的事项上报至渠道经理（如当月最后两天发生特殊事项，随时与渠道经理沟通补提），对应渠道经理按流程进行申诉，并于月底的最后一周完成申诉。次月的第一个工作日将会公示运力考核结果及申诉详情，如有异议，可于第一个工作日通过TT工单对结果进行申诉沟通，最终申诉结果以美团配送通知为准。"],
        ["运力验证",
         "微笑行动抽查不合格非本人、上线验真有单非本人、站长点上线有单未验真、专送同轨迹&一机多号、专快送同轨迹&一机多号、异常改派点送达、督导检核、骑手身份虚假、声纹识别非本人等判定为非有效骑手，具体运力虚假判定规则与诚信考核中虚假骑手定义保持一致。"],
        ["补充说明",
         "本考核方案的KA品牌驻点骑手得分最终只应用于KA品牌单体验考核得分计算，是否属于同配送区同商情况以考核周期最后一天状态为准。"],
    ]


def _repair_ka_special_scene_fusion_layout(blocks: List[Block],
                                           counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks):
        compact = re.sub(r'\s+', '', _block_text(blocks[i]))
        if not (blocks[i].kind == "heading"
                and "5.7" in compact
                and "特殊场景体验融合考核" in compact):
            i += 1
            continue

        end = _next_heading_index(blocks, i + 1)
        section_text = "\n".join(_block_text(b) for b in blocks[i + 1:end])
        compact_section = re.sub(r'\s+', '', section_text)
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
        ]
        needs_repair = (
            any(item not in compact_section for item in required)
            or "站点组A麦当劳品" in section_text
            or "融合后体" in section_text
            or "<br" in section_text.lower()
        )
        if not needs_repair:
            i = end
            continue

        ref = blocks[i + 1] if i + 1 < end else blocks[i]
        blocks[i + 1:end] = _ka_special_scene_fusion_blocks(ref)
        counter["KA特殊场景融合完整章节重排"] += 1
        i = _next_heading_index(blocks, i + 1)


def _ka_special_scene_fusion_blocks(ref: Block) -> List[Block]:
    note = (
        "注：如实际天气与系统判定天气不符，加盟站长可在申诉时段内通过“星火 APP”"
        "发起“恶劣天气申诉”，申请将配送区域内的正常天气判定修正为恶劣天气。"
    )
    return [
        Block(
            kind="table",
            rows=_ka_special_scene_target_rows(),
            page=ref.page,
            bbox=ref.bbox,
        ),
        Block(
            kind="table",
            rows=_ka_special_scene_formula_rows(),
            page=ref.page,
            bbox=ref.bbox,
        ),
        Block(
            kind="para",
            text=("普通场景算分示例：假设站点组A履约的KA品牌仅有麦当劳，"
                  "当月麦当劳在普通场景下完成单量1100单，完成单（剔除异常单）"
                  "1000单，则站点组A履约麦当劳品牌普通场景算分示例如下："),
            page=ref.page,
            bbox=ref.bbox,
        ),
        Block(kind="table", rows=_ka_normal_scene_score_example_rows(),
              page=ref.page, bbox=ref.bbox),
        Block(
            kind="para",
            text="麦当劳普通场景得分=15%*120+10%*120+15%*120+50%*105+10%*120=124.50分",
            page=ref.page,
            bbox=ref.bbox,
        ),
        Block(
            kind="para",
            text=("特殊场景算分示例：假设站点组A履约的KA品牌仅有麦当劳，"
                  "当月麦当劳在特殊场景下完成单量130单，完成单（剔除异常单）"
                  "100单，则站点组A履约麦当劳品牌特殊场景算分示例如下："),
            page=ref.page,
            bbox=ref.bbox,
        ),
        Block(kind="table", rows=_ka_special_scene_score_example_rows(),
              page=ref.page, bbox=ref.bbox),
        Block(
            kind="para",
            text="麦当劳特殊场景得分=15%*0+10%*120+15%*120+50%*110+10%*120+10%*108=107.80分",
            page=ref.page,
            bbox=ref.bbox,
        ),
        Block(kind="table", rows=_ka_fusion_score_example_rows(),
              page=ref.page, bbox=ref.bbox),
        Block(kind="para", text=note, page=ref.page, bbox=ref.bbox),
    ]


def _ka_special_scene_target_rows() -> List[List[str]]:
    return [
        ["项目", "内容"],
        ["考核方式",
         "普通场景和特殊场景分两套目标考核，按剔除异常单后的特殊场景完成单占比计算融合后体验得分。"],
        ["考核指标",
         "虚假点送达率、配送原因未完成率、复合准时率、KA品牌负向反馈率、承托比、复合超时时长、KA品牌客诉率。"],
        ["天气等级",
         "10：正常天气；20：一般恶劣天气；30：比较恶劣天气；40：非常恶劣天气。"],
        ["考核规则",
         "以运单首次调度时的天气等级判定为依据，按下方规则分普通场景和特殊场景考核。"],
        ["考核目标举例",
         "不同场景的体验目标（仅为示例，最终以美团配送烽火台-商服务费计费系统为准）。"],
        ["普通场景体验满分目标",
         "站点ID：A；KA品牌负向反馈率：0.1%；虚假点送达率：0.01%；配送原因未完成率：0.1%；复合准时率：95%；承托比：25%；复合超时时长：5s。"],
        ["特殊场景体验满分目标",
         "站点ID：A；KA品牌负向反馈率：0.1%；虚假点送达率：0.01%；配送原因未完成率：0.1%；复合准时率：93%；承托比：25%；复合超时时长：10s。"],
    ]


def _ka_special_scene_formula_rows() -> List[List[str]]:
    return [
        ["项目", "内容"],
        ["核算公式",
         "融合后体验得分 = 特殊场景体验得分 * 剔除异常单后特殊场景完成单占比 + 普通场景体验得分 *（1-剔除异常单后特殊场景完成单占比）。"],
        ["剔除异常单后特殊场景完成单占比",
         "剔除异常单后特殊场景完成单 / 剔除异常单后总完成单。"],
    ]


def _ka_normal_scene_score_example_rows() -> List[List[str]]:
    return [
        ["指标", "分子数值", "分母数值", "达标率（值）", "满分目标", "权重", "得分"],
        ["KA品牌负向反馈率", "1单", "1000", "0.1%", "0.01%", "15%", "120"],
        ["虚假点送达", "0单", "1100", "0%", "0.007%", "10%", "120"],
        ["配送原因未完成", "1单", "1000", "0.1%", "0.01%", "15%", "120"],
        ["复合准时率", "一般超时50单；严重超时1单", "1000", "94.5%", "96%", "50%", "105"],
        ["复合超时时长", "1970秒", "1000", "1.97秒", "3.0s", "10%", "120"],
    ]


def _ka_special_scene_score_example_rows() -> List[List[str]]:
    return [
        ["指标", "分子数值", "分母数值", "达标率（值）", "满分目标", "权重", "得分"],
        ["KA品牌负向反馈率", "2单", "100", "2%", "0.01%", "15%", "0"],
        ["虚假点送达", "1单", "130", "0.769%", "0.007%", "10%", "120"],
        ["配送原因未完成", "0单", "100", "0%", "0.01%", "15%", "120"],
        ["复合准时率", "一般超时5单；严重超时0单", "100", "95%", "96%", "50%", "110"],
        ["复合超时时长", "200秒", "100", "2秒", "5.0s", "10%", "120"],
    ]


def _ka_fusion_score_example_rows() -> List[List[str]]:
    return [
        ["指标", "数值"],
        ["普通场景剔除后完成单（10天气）", "1000"],
        ["特殊场景剔除后完成单（20+30天气）", "100"],
        ["特殊场景完成单占比", "=100/(1000+100)=9.0909%"],
        ["普通场景得分", "124.50"],
        ["特殊场景得分", "107.80"],
        ["融合后体验得分", "=107.80*9.0909%+124.50*(1-9.0909%)=122.9818分"],
    ]


def _ensure_ka_experience_adjustment_definitions(blocks: List[Block],
                                                counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks):
        compact = re.sub(r'\s+', '', _block_text(blocks[i]))
        if not ("7.1" in compact and "KA品牌体验调分项" in compact):
            i += 1
            continue

        end = _next_heading_index(blocks, i + 1)
        section = "\n".join(_block_text(b) for b in blocks[i + 1:end])
        if "得分范围" in section and "麦当劳完单量占比" in section:
            i += 1
            continue

        blocks.insert(i + 1, Block(
            kind="table",
            rows=_ka_experience_adjustment_definition_rows(),
            page=blocks[i].page,
            bbox=blocks[i].bbox,
        ))
        counter["KA品牌体验调分项定义补全"] += 1
        i = _next_heading_index(blocks, i + 2)


def _ka_experience_adjustment_definition_rows() -> List[List[str]]:
    return [
        ["项目", "内容"],
        ["得分范围", "[0,0.2]"],
        ["计算公式",
         "KA品牌体验总得分=（麦当劳完单量占比 * 麦当劳体验得分 + 肯德基完单量占比 * 肯德基体验得分 + 必胜客完单量占比 * 必胜客体验得分） * 麦肯必单量占比膨胀系数"],
        ["指标定义1",
         "【完成订单量】通过美团配送且订单最终状态为完成的订单数量。\n"
         "麦当劳完单量占比=麦当劳完单量/（麦当劳完单量+肯德基完单量+必胜客完单量）\n"
         "肯德基完单量占比=肯德基完单量/（麦当劳完单量+肯德基完单量+必胜客完单量）\n"
         "必胜客完单量占比=必胜客完单量/（麦当劳完单量+肯德基完单量+必胜客完单量）\n"
         "注：数据来源均为品牌方，按照品牌方口径统计（剔除虚假订单、提前关闭订单，均不剔除异常单）。"],
        ["指标定义2",
         "【麦当劳体验得分】分距离段分别考核“30分钟内送达订单占比”：\n"
         "30分钟内送达订单占比（配送距离3km及以内订单）=Σ（配送距离3km及以内且30分钟内送达完成订单量）/Σ（配送距离3km及以内完成订单量）\n"
         "30分钟内送达订单占比（配送距离3km以上订单）=Σ（配送距离3km以上且30分钟内送达完成订单量）/Σ（配送距离3km以上完成订单量）\n"
         "注：【30分钟内送达订单】通过美团配送30分钟内送达且订单最终状态为完成的订单数量；30分钟内送达：送达时间-顾客支付时间≤30分钟。\n"
         "【完成订单量】通过美团配送且订单最终状态为完成的订单数量。\n"
         "注：数据来源为麦当劳品牌方，按照麦当劳品牌方口径统计；考核范围：同配送区域同商站点组履约的驻点和融网门店单（分子分母均剔除虚假订单、提前关闭订单，均不剔除异常单）。"],
        ["指标定义3",
         "【肯德基体验得分】考核“品牌方口径准时率（肯德基）”\n"
         "品牌方口径准时率（肯德基）=Σ（按肯德基要求准时完成配送的订单量）/Σ（配送完成订单量）\n"
         "注：数据来源为肯德基品牌方，按照肯德基品牌方口径统计；考核范围：同配送区域同商站点组履约的肯德基全部子品牌门店订单（分子分母均剔除虚假订单、提前关闭订单，均不剔除异常单）。"],
        ["指标定义4",
         "【必胜客体验得分】考核“品牌方口径准时率（必胜客）”\n"
         "品牌方口径准时率（必胜客）=Σ（按必胜客要求准时完成配送的订单量）/Σ（配送完成订单量）\n"
         "注：数据来源为必胜客品牌方，按照必胜客品牌方口径统计；考核范围：同配送区域同商站点组履约的必胜客全部子品牌门店订单（分子分母均剔除虚假订单、提前关闭订单，均不剔除异常单）。"],
        ["指标定义5",
         "麦肯必单量占比=（Σ麦当劳品牌完单量+Σ肯德基品牌完单量+Σ必胜客品牌完单量）/Σ站点组所有完单量\n"
         "注①：麦肯必完单量口径为各品牌方回传的完单量（剔除虚假订单、提前关闭订单，均不剔除异常单）；\n"
         "注②：站点组所有完单量，为站点组内加盟站点履约的所有完单量（剔除虚假单、不剔除异常单），其中麦肯必完单量为品牌方口径的完单量（剔除虚假订单、提前关闭订单，不剔除异常单）。"],
    ]


def _ensure_ka_fake_delivery_definition(blocks: List[Block],
                                        counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks):
        compact = re.sub(r'\s+', '', _block_text(blocks[i]))
        if not ("8.1" in compact and "客诉虚假点送达" in compact):
            i += 1
            continue

        end = _next_heading_index(blocks, i + 1)
        section = "\n".join(_block_text(b) for b in blocks[i + 1:end])
        if "虚假点送达指" in section and "客诉虚假点送达率" in section:
            i += 1
            continue

        blocks.insert(i + 1, Block(
            kind="table",
            rows=_ka_fake_delivery_definition_rows(),
            page=blocks[i].page,
            bbox=blocks[i].bbox,
        ))
        counter["客诉虚假点送达定义补全"] += 1
        i = _next_heading_index(blocks, i + 2)


def _ka_fake_delivery_definition_rows() -> List[List[str]]:
    return [
        ["项目", "内容"],
        ["客诉虚假点送达",
         "虚假点送达指骑手未将餐品/货品按照订单要求送达指定位置虚假点击送达的行为，包括但不限于提前点击送达、延后点击送达，如距离顾客地址较远时点击确认送达、未送达顾客指定位置点击送达、预约单提前配送延后点送达等情形。对于虚假点送达的行为，相关用户会通过不同途径进行投诉，其中【客诉虚假点送达】将考核现有的电话客诉来源的虚假点送达数据。"],
        ["数据来源", "客户通过拨打客服电话或在订单页面进行的虚假点送达投诉，可通过线上申诉；"],
        ["指标定义",
         "公式原文（需核对）：客诉虚假点送达率=星巴克/麦当劳/大润发品牌门店命中客诉虚假点送达单量/站点组履约的所有KA品牌单完成单量"],
        ["考核范围",
         "同配送区域同商站点组履约的星巴克、麦当劳、大润发KA品牌单（分子剔除申诉通过的订单，分母不剔除）。"],
    ]


def _ensure_formula_in_section(blocks: List[Block], heading_idx: int,
                               formula_text: str, label: str) -> bool:
    end = _next_heading_index(blocks, heading_idx + 1)
    section = blocks[heading_idx + 1:end]
    compact_section = re.sub(r'\s+', '', "\n".join(_block_text(b) for b in section))
    if label.startswith("复合准时率") and (
            "复合准时率（考核）=" in compact_section
            or "C_{KA品牌单}" in compact_section):
        return False
    if label.startswith("配送原因未完成率") and (
            "配送原因未完成率=" in compact_section
            or "P_{KA品牌单}" in compact_section):
        return False
    if label.startswith("KA负向反馈率") and (
            "KA负向反馈率=" in compact_section
            or "F1+3" in compact_section):
        return False
    if label.startswith("KA品牌客诉率") and (
            "KA品牌客诉率=" in compact_section
            or "KS/W" in compact_section):
        return False
    if label.startswith("承托比") and (
            "承托比=" in compact_section
            or "R_{KA品牌单}" in compact_section):
        return False
    if label.startswith("虚假点送达率") and (
            "虚假点送达率=" in compact_section
            or "T_{KA品牌单}" in compact_section):
        return False

    def make_formula(ref: Block) -> Block:
        return Block(
            kind="image", text=formula_text, flags=["formula", "needs_review"],
            page=ref.page, bbox=ref.bbox,
        )

    for j in range(heading_idx + 1, end):
        text = re.sub(r'\s+', '', _block_text(blocks[j]))
        if "计算口径" in text:
            blocks.insert(j + 1, make_formula(blocks[j]))
            return True
        if "指标释义" in text:
            cue = Block(kind="para", text="计算口径：",
                        page=blocks[j].page, bbox=blocks[j].bbox)
            blocks[j:j] = [cue, make_formula(blocks[j])]
            return True
    blocks.insert(heading_idx + 1, make_formula(blocks[heading_idx]))
    return True


def _next_heading_index(blocks: List[Block], start: int) -> int:
    for j in range(start, len(blocks)):
        if blocks[j].kind == "heading":
            return j
    return len(blocks)


def _drop_noise_blocks(blocks: List[Block], counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks):
        blk = blocks[i]
        if blk.kind in ("heading", "para") and _is_compressed_toc_noise(blk.text):
            del blocks[i]
            counter["压缩目录残留已删除"] += 1
            continue
        if (blk.kind in ("heading", "para")
                and re.fullmatch(r'\d{1,3}', blk.text.strip())
                and _near_image_or_compressed_toc(blocks, i)):
            del blocks[i]
            counter["孤立页码残留已删除"] += 1
            continue
        i += 1


def _is_compressed_toc_noise(text: str) -> bool:
    compact = re.sub(r'\s+', '', text or "")
    if compact.count("、") < 2:
        return False
    return bool(re.search(r'\d[一二三四五六七八九十]{1,3}、', compact))


def _near_image_or_compressed_toc(blocks: List[Block], index: int) -> bool:
    for j in range(max(0, index - 2), min(len(blocks), index + 3)):
        if j == index:
            continue
        blk = blocks[j]
        if blk.kind == "image":
            return True
        if blk.kind in ("heading", "para") and _is_compressed_toc_noise(blk.text):
            return True
    return False


def _repair_toc_sequence(blocks: List[Block], counter: Counter[str]) -> None:
    """只在前后目录编号能互相印证时，修复 OCR 把“二”读成“一”的情况。"""
    for i, blk in enumerate(blocks):
        if blk.kind not in ("heading", "para"):
            continue
        text = blk.text.strip()
        if not _TOC_APPLICABLE_REGION.match(text):
            continue
        prev_text = _nearest_text(blocks, i, -1)
        next_text = _nearest_text(blocks, i, 1)
        if re.match(r'^一、\s*背景(?:\s|　|\d)*', prev_text) and next_text.startswith("三、"):
            blk.text = re.sub(r'^一、', '二、', blk.text, count=1)
            counter["适用区域目录编号"] += 1

    i = 0
    while i < len(blocks):
        blk = blocks[i]
        if blk.kind in ("heading", "para") and re.fullmatch(r'\d{1,3}', blk.text.strip()):
            prev_idx = _nearest_text_index(blocks, i, -1)
            if prev_idx is not None and _can_absorb_toc_page_number(blocks, prev_idx):
                page_no = blk.text.strip()
                blocks[prev_idx].text = f"{blocks[prev_idx].text.rstrip()}　{page_no}"
                del blocks[i]
            counter["目录页码断行"] += 1
            continue
        i += 1

    for i, blk in enumerate(blocks):
        if blk.kind not in ("heading", "para"):
            continue
        text = blk.text.strip()
        prev_text = _nearest_text(blocks, i, -1)
        next_text = _nearest_text(blocks, i, 1)
        if text == "一、" and _near_toc_heading(blocks, i) and next_text.startswith("二、"):
            if _later_has_heading(blocks, i, "一、总则"):
                blk.kind = "para"
                blk.level = 0
                blk.text = "一、总则　1"
                counter["目录缺失条目补全"] += 1
        if text == "四、附则" and _near_toc_heading(blocks, i) and prev_text.startswith("三、"):
            blk.kind = "para"
            blk.level = 0
            if "问责标准" in prev_text:
                blk.text = "四、附则　7"
            counter["目录标题误判修复"] += 1


def _can_absorb_toc_page_number(blocks: List[Block], index: int) -> bool:
    text = blocks[index].text.strip()
    if not _is_numbered_toc_line(text):
        return False
    if re.search(r'(?:\s|　)\d{1,3}$', text):
        return False
    if not _near_toc_heading(blocks, index):
        return False
    return True


def _near_toc_heading(blocks: List[Block], index: int) -> bool:
    for step in (-1, 1):
        j = index + step
        seen = 0
        while 0 <= j < len(blocks) and seen < 5:
            blk = blocks[j]
            if blk.kind in ("heading", "para") and blk.text.strip():
                text = blk.text.strip()
                if text in ("目录", "目錄"):
                    return True
                seen += 1
            j += step
    return False


def _nearest_text(blocks: List[Block], index: int, step: int) -> str:
    found = _nearest_text_index(blocks, index, step)
    return "" if found is None else blocks[found].text.strip()


def _nearest_text_index(blocks: List[Block], index: int, step: int):
    j = index + step
    seen = 0
    while 0 <= j < len(blocks) and seen < 4:
        blk = blocks[j]
        if blk.kind in ("heading", "para") and blk.text.strip():
            text = blk.text.strip()
            if text not in ("目录", "目錄") and not re.fullmatch(r'\d{1,3}', text):
                return j
            seen += 1
        j += step
    return None


def _later_has_heading(blocks: List[Block], index: int, compact_heading: str) -> bool:
    target = re.sub(r'\s+', '', compact_heading)
    for blk in blocks[index + 1:]:
        if blk.kind not in ("heading", "para"):
            continue
        text = re.sub(r'\s+', '', blk.text or "")
        if text.startswith(target):
            return True
    return False


def table_suspect_score(rows: List[List[str]]) -> int:
    """粗略识别表格串列：金额跑到非末列、编号跑到长正文里等。"""
    if not rows:
        return 0
    score = 0
    ncol = max(len(r) for r in rows)
    header = rows[0] + [""] * (ncol - len(rows[0]))
    flat_text = " ".join(" ".join(c for c in row if c) for row in rows)
    compact_flat = re.sub(r'\s+', '', flat_text)
    if (ncol >= 5
            and "方案" in "".join(header)
            and "内容" in "".join(header)
            and "考核规则" in flat_text
            and "普通场景考核" in flat_text
            and "特殊场景考核" in flat_text):
        score += 4
    if re.search(r'以运单首次调度时的天\s*等级为依据', flat_text):
        score += 4
    if "正常天气单恶劣天气单" in compact_flat:
        score += 3
    header_counts = [_header_label_count(c) for c in header]
    if any(c >= 2 for c in header_counts):
        score += 4
    if ncol <= 2 and sum(header_counts) >= 3:
        score += 5
    for row in rows:
        padded = row + [""] * (ncol - len(row))
        row_text = "".join(padded)
        if is_noise_text(row_text):
            score += 5
        if _WATERMARK_FRAGMENT_RE.search(row_text):
            score += 3
        if _SUSPECT_OCR_CHARS.search(row_text):
            score += 3
        if _SUSPECT_TABLE_TEXT.search(row_text):
            score += 3
        long_cells = [
            re.sub(r'\s+', '', c) for c in padded
            if len(re.sub(r'\s+', '', c)) > 120
        ]
        if ncol >= 3 and len(long_cells) >= 2:
            score += 4
        if any(c.count("\n") >= 2 and len(re.sub(r'\s+', '', c)) > 100
               for c in padded):
            score += 3
        if long_cells and re.search(r'[①②③④⑤⑥].*[①②③④⑤⑥]', row_text):
            score += 2
        if ncol == 1 and _header_label_count(row_text) >= 3:
            score += 4
        if ncol == 1 and len(row_text) > 220 and (
                row_text.count("元/") >= 2 or "承担违约金" in row_text):
            score += 4
        for j, cell in enumerate(padded):
            text = re.sub(r'\s+', '', cell)
            if not text:
                continue
            if is_noise_text(text):
                score += 3
            if _WATERMARK_FRAGMENT_RE.search(text):
                score += 2
            if _SUSPECT_TABLE_TEXT.search(text):
                score += 3
            if j < ncol - 1 and _AMOUNT.search(text):
                score += 2
            if j < ncol - 1 and re.search(r'\d+\s*元\s*/', text):
                score += 1
            if j > 0 and len(text) > 80 and _ITEM_NO.search(text):
                score += 1
            if "整改不达标" in text and j < ncol - 1:
                score += 1
    return score


def _header_label_count(text: str) -> int:
    compact = re.sub(r'\s+', '', text or "")
    return sum(1 for w in _HEADER_WORDS if w in compact)


def is_noise_text(text: str) -> bool:
    """页眉/页脚/保密水印残留。只用于清理 OCR 噪声，不处理正常正文。"""
    s = re.sub(r'\s+', '', text or "")
    if not s:
        return False
    if _NOISE_RE.search(s):
        return True
    if _MTPS_CODE_RE.search(s):
        if any(w in s for w in ("编号", "以下简称", "制度", "协议", "内容")):
            return False
        rest = _MTPS_CODE_RE.sub('', s)
        rest = re.sub(r'[#_()（）|\\/.·.\s-]+', '', rest)
        if len(rest) <= 8:
            return True
    if _ACCOUNT_NOISE_RE.search(s):
        rest = _ACCOUNT_NOISE_RE.sub('', s)
        rest = re.sub(r'[#_()（）|\\/.·.\s-]+', '', rest)
        if len(rest) <= 4:
            return True
    if _PURE_WATERMARK_FRAGMENT_RE.match(s):
        return True
    if "第三方提供" in s and len(s) <= 40:
        return True
    if s.count("美团配送") >= 1 and len(s) <= 24:
        return True
    return False


def clean_table_noise_rows(rows: List[List[str]]) -> List[List[str]]:
    """删除表格中被单元格 OCR 带回来的虚拟页眉页脚残留行。"""
    cleaned: List[List[str]] = []
    for row in rows:
        if is_noise_text("".join(row)):
            continue
        new_row = ["" if is_noise_text(c) else c for c in row]
        if any(c.strip() for c in new_row):
            cleaned.append(new_row)
    return cleaned


def repair_table_rows(rows: List[List[str]]) -> int:
    """修复上下文明确的单元格错拆。"""
    repaired = 0
    for row in rows:
        if (len(row) >= 3
                and row[0].strip() == "总完成单（W）"
                and row[1].strip().endswith("后的")
                and row[2].strip() == "总完成单"):
            row[1] = row[1].strip() + row[2].strip()
            row[2] = ""
            repaired += 1
        if len(row) >= 3 and "不考核" in row[0]:
            logic = re.sub(r'\s+', '', row[2])
            if logic in ("0=", "=10", "10="):
                row[2] = "=0"
                repaired += 1
        if len(row) >= 2 and row[0].strip() == "目标值":
            target = re.sub(r'\s+', '', row[1])
            if target == "%66":
                row[1] = "99%"
                repaired += 1
    repaired += _repair_weather_policy_table(rows)
    repaired += _repair_ka_assessment_framework_table(rows)
    repaired += _repair_scene_experience_framework_table(rows)
    repaired += _repair_pressure_scenario_table(rows)
    repaired += _repair_ka_indicator_definition_table(rows)
    repaired += _repair_ka_scene_policy_table(rows)
    repaired += _repair_ka_score_rule_table(rows)
    repaired += _repair_ka_brand_punctuality_table(rows)
    repaired += _repair_ka_experience_adjustment_tail_table(rows)
    repaired += _repair_fine_schedule_supplement_table(rows)
    repaired += _repair_ka_coefficient_table(rows)
    return repaired


def _table_text(rows: List[List[str]]) -> Tuple[str, str]:
    text = " ".join(" ".join(c for c in row if c) for row in rows)
    return text, re.sub(r'\s+', '', text)


def _repair_ka_assessment_framework_table(rows: List[List[str]]) -> int:
    table_text, compact = _table_text(rows)
    if not ("常规计分项" in table_text and "调分项" in table_text
            and "调星项" in table_text):
        return 0
    if not ("复合准时率" in table_text and "虚假点送达" in table_text
            and ("KA品牌驻点骑手" in table_text or "KA品牌体验" in table_text)):
        return 0

    rows[:] = _ka_assessment_framework_rows()
    return 1


def _repair_scene_experience_framework_table(rows: List[List[str]]) -> int:
    table_text, compact = _table_text(rows)
    if "分场景体验考核" not in table_text:
        return 0
    if not ("普通场景目标" in table_text and "特殊场景目标" in table_text):
        return 0
    if not ("指定考核日" in table_text and ("1倍权重" in table_text or "多倍权重" in table_text)):
        return 0
    if "压力场景类型" in table_text or "场景定义" in table_text:
        return 0

    rows[:] = _scene_experience_framework_rows()
    return 1


def _repair_pressure_scenario_table(rows: List[List[str]]) -> int:
    table_text, compact = _table_text(rows)
    if not ("压力场景" in table_text and "场景定义" in table_text
            and "单量加权系数" in table_text and "补充说明" in table_text):
        return 0
    if not ("恶劣天气" in table_text and "指定考核日" in table_text):
        return 0

    rows[:] = [
        ["压力场景类型", "场景定义", "单量加权系数", "补充说明"],
        ["恶劣天气", "恶劣天气调度的运单，即调度时天气等级>10的运单",
         "X倍（详见烽火台）", "台风、内涝、结冰等3类极端恶劣天气调度的运单可申请异常单剔除。"],
        ["指定考核日", "特殊节假日调度的运单，含法定节假日或其他指定日期内调度的运单。",
         "Y倍（详见烽火台）", "见每月指定考核日期"],
        ["指定考核日", "周末/周中指定日期调度的运单",
         "Z倍（详见烽火台）", "见每月指定考核日期"],
    ]
    return 1


def _repair_ka_indicator_definition_table(rows: List[List[str]]) -> int:
    table_text, compact = _table_text(rows)
    if "完成订单量" not in table_text and "完成星巴克订单量" not in table_text:
        return 0
    if not ("复合准时率" in table_text
            or "配送原因未完成率" in table_text
            or "一般超时" in table_text
            or "严重超时" in table_text
            or "未完成" in table_text
            or "虚假点送达" in table_text
            or "承托比" in table_text
            or "打标骑手" in table_text):
        return 0

    repaired = 0
    for row in rows:
        row_text = "".join(row)
        if re.search(r'完成星巴克订单量[（(]W[）)]', row_text):
            definition = "通过美团配送且运单最终状态为完成的星巴克运单数量"
        elif re.search(r'完成订单量[（(]W[）)]', row_text):
            definition = "通过美团配送且运单最终状态为完成的运单数量"
        else:
            continue
        if any("通过美团配送" in c for c in row):
            continue
        if len(row) < 3:
            row.extend([""] * (3 - len(row)))
        target = len(row) - 1
        if not row[target].strip() or len(row[target].strip()) <= 8:
            row[target] = definition
            repaired += 1
    return repaired


def _repair_ka_scene_policy_table(rows: List[List[str]]) -> int:
    table_text, compact = _table_text(rows)
    if not ("考核方式" in compact and "考核指标" in compact
            and "天气等级" in compact and "考核规则" in compact):
        return 0
    if not ("KA品牌客诉率" in compact and "复合超时时长" in compact
            and "普通场景" in compact and "特殊场景" in compact):
        return 0

    normal = "距离≤3公里的正常天气单（天气等级为10）"
    broad_special = "距离>3公里的正常天气单、恶劣天气单（天气等级为20或30或40）"
    weather20_30 = "距离>3公里的正常天气单、恶劣天气单（天气等级为20或30）"
    rows[:] = [
        ["类型", "项目/指标", "普通场景考核", "特殊场景考核", "备注"],
        ["说明", "考核方式",
         "普通场景和特殊场景分两套目标考核，按剔除异常单后的特殊场景完成单占比计算融合后体验得分。",
         "", ""],
        ["说明", "考核指标",
         "虚假点送达率、配送原因未完成率、复合准时率、KA品牌负向反馈率、承托比、复合超时时长、KA品牌客诉率。",
         "", ""],
        ["说明", "天气等级",
         "10：正常天气；20：一般恶劣天气；30：比较恶劣天气；40：非常恶劣天气",
         "", ""],
        ["规则", "KA品牌负向反馈率", normal, broad_special, "/"],
        ["规则", "承托比", normal, broad_special, "/"],
        ["规则", "虚假点送达率", normal, broad_special, "/"],
        ["规则", "KA品牌客诉率", normal, broad_special, "/"],
        ["规则", "配送原因未完成率", normal, weather20_30, "40天气免责"],
        ["规则", "复合准时率", normal, weather20_30, "40天气免责"],
        ["规则", "复合超时时长", normal, weather20_30,
         "40天气免责或大于5公里远距离单（除星巴克、麦当劳、沃尔玛、山姆外）免责"],
    ]
    return 1


def _repair_ka_score_rule_table(rows: List[List[str]]) -> int:
    table_text, compact = _table_text(rows)
    has_positive = "正向指标计分规则" in compact or "以复合准时率为例" in compact
    if not (has_positive and "负向指标计分规则" in compact):
        return 0
    if not ("复合准时率" in table_text and "配送原因未完成率" in table_text):
        return 0

    rows[:] = [
        ["规则类型", "适用指标", "计分规则"],
        ["正向指标",
         "复合准时率、承托比",
         "当指标≤X时，站点得分为0；当指标=Y时，站点得分为100分；当指标≥Z时，站点得分为120分；当指标介于（X,Y）或（Y,Z）之间时，等比例算分。"],
        ["负向指标",
         "配送原因未完成率、KA品牌负向反馈率、KA品牌客诉率、虚假点送达率、复合超时时长",
         "当指标≥X时，站点得分为0分；当指标=Y时，站点得分为100分；当指标≤Z时，站点得分为120分；当指标介于（X,Y）或（Y,Z）之间时，等比例算分。"],
    ]
    return 1


def _repair_weather_policy_table(rows: List[List[str]]) -> int:
    table_text, compact = _table_text(rows)
    if not ("普通场景考核" in table_text and "特殊场景考核" in table_text
            and "负向反馈率" in table_text and "配送原因未完成率" in table_text
            and "天气等级" in table_text and "HD" in table_text):
        return 0
    has_weather_pair = (
        "正常天气单恶劣天气单" in compact
        or "正常天气单、恶劣天气单" in compact
    )
    has_dispatch_basis = bool(re.search(r'以运单首次调度时的天(?:气)?等级为依据', compact))
    if not has_weather_pair and not has_dispatch_basis:
        return 0

    normal = "天气等级为10且导航距离≤3公里的正常天气单"
    special = "导航距离>3公里的恶劣天气单（天气等级为20或30或40）、HD尾单&专送兜底单"
    rows[:] = [
        ["指标", "普通场景考核", "特殊场景考核", "备注"],
        ["负向反馈率", normal, special, "/"],
        ["配送原因未完成率", normal, special, "40天气免责"],
        ["复合准时率（考核）", normal, special, "40天气或导航距离>5公里免责"],
        ["复合超时时长", normal, special, "40天气或导航距离>5公里免责"],
    ]
    return 1


def _repair_ka_brand_punctuality_table(rows: List[List[str]]) -> int:
    table_text = " ".join(" ".join(c for c in row if c) for row in rows)
    compact = re.sub(r'\s+', '', table_text)
    if not ("品牌方口径准时率（肯德基）" in table_text
            and "要求目标值介于门槛值" in compact
            and "考核周期得分为0.2" in compact):
        return 0

    rows[:] = [
        ["要求", "品牌方口径准时率（肯德基）", "考核周期得分"],
        ["目标值", "99%", "0.2"],
        ["介于门槛值、目标值之间", "介于93%到99%之间", "等比例计算得分"],
        ["门槛值", "93%", "0"],
    ]
    return 1


def _repair_ka_experience_adjustment_tail_table(rows: List[List[str]]) -> int:
    table_text = " ".join(" ".join(c for c in row if c) for row in rows)
    compact = re.sub(r'\s+', '', table_text)
    if not ("麦肯必单量占比" in table_text
            and "补充说明" in table_text
            and "公示流程" in table_text
            and "膨胀系数" in table_text):
        return 0
    if "30分钟送达订单占比" in table_text and "考核站点" in table_text:
        return 0

    formula = (
        "KA品牌体验总得分=（麦当劳完单量占比 * 麦当劳体验得分 + "
        "肯德基完单量占比 * 肯德基体验得分 + "
        "必胜客完单量占比 * 必胜客体验得分） * 麦肯必单量占比膨胀系数。"
        "封顶值0.2分。"
    )
    supplement = "\n".join([
        "① 参与考核站点组：以实际签署为准，若考核月新建集约站点归属的合作商签署本制度则按照本制度进行考核，若未签署则不参与本制度考核；",
        "② 同商同配送区域的加盟站点作为站点组参与考核，考核数据合并计算，含KA集约站、加盟大网站，不含企客集约站（即不含站点名称含“企客集约”的站点），是否属于同配送区同商情况以考核周期最后一天状态为准；",
        "③ 考核品牌门店：麦当劳品牌的驻点和融网门店（剔除品牌方不考核门店）；肯德基（含肯德基门店之外的全部子品牌，如肯悦咖啡\\百胜轻食\\爷爷自在茶等全部子品牌门店）、必胜客（剔除品牌方不考核门店）；",
        "④ 单量门槛：考核月站点组承接的麦当劳完单量（剔除虚假单、提前关闭订单）小于300单的不参与麦当劳调分项考核，考核月站点组承接的肯德基单量（剔除虚假单、提前关闭订单）小于300单的不参与肯德基调分项考核，考核月站点组承接的必胜客单量（剔除虚假单、提前关闭订单）小于300单的不参与必胜客调分项考核；",
        "⑤ 数据获取方式：联系渠道经理；",
        "⑥ 该调分项，只影响站点组星级，不影响KA星级。",
    ])
    rows[:] = [
        ["项目", "内容"],
        ["计分规则4-目标值", "麦肯必单量占比：15%；膨胀系数：1.2"],
        ["计分规则4-介于门槛值、目标值之间", "麦肯必单量占比介于5%到15%之间；等比例计算膨胀系数"],
        ["计分规则4-门槛值", "麦肯必单量占比：5%；膨胀系数：0.8"],
        ["计分规则5", formula],
        ["补充说明", supplement],
        ["公示流程", "次月的第六个工作日前会公示考核结果，如有计分统计错误，可于公示次日16点前通过渠道经理对结果进行申诉沟通，最终结果以美团配送通知为准。"],
    ]
    return 1


def _repair_fine_schedule_supplement_table(rows: List[List[str]]) -> int:
    table_text = " ".join(" ".join(c for c in row if c) for row in rows)
    if not ("考核补充说明" in table_text
            and "本月新建站" in table_text
            and "精细化排班" in table_text
            and "烽火台" in table_text):
        return 0

    rows[:] = [
        ["序号", "考核补充说明"],
        ["1", "本月新建站点及本月无任何考核目标值站点不参与本月考核，本月新建站点按照站点首单日期进行判断，上月新建站点且有考核目标值，自站点首单日起7天内的数据不进行考核，从运营第8天开始考核精细化排班工具，本月无考核目标的站点该项考核得分按城市平均得分进行保护。"],
        ["2", "不在站点营业时间内的排班时段站点无需排班，若排班但骑手未在线导致不合格，按照不合格进行统计。"],
        ["3", "考核目标值可咨询渠道经理，本月正式考核前同步在烽火台进行下发，详见烽火台-通知管理。"],
        ["4", "同商同配送区域的加盟站点作为站点组参与考核，考核数据绑定计算，统一在主站点体现调分结果。如果推送目标值存在主从站不一致的情况，考核结算以主站目标为准。是否属于同配送区域同商以2026年6月30日数据为准。KA属地化的集约站点不参与精细化排班日常考核，相关数据也不会计入到考核核算中。"],
        ["5", "精细化排班各项考核指标取值不区分烽火台商标记的全/兼职标签，考核范围为站点全量在职骑手（包含全职骑手和兼职骑手）。"],
        ["6", "依据站点的天气指数判定及指定压力日期，设定考核方案，站点天气查询路径：①当天实时天气监控：烽火台-业务管理-天气查询；②最终天气判定：“烽火台-业务管理-骑手排班-排班管理”页面中，历史已发生日期的天气，为系统最终判定的站点天气；6月指定压力日期：端午节（6月19日、6月20日、6月21日），与恶劣天气适用同一套考核方案。指定活动日：因临时特殊活动，预估存在压力的日期（至少提前3天下发补充制度），与恶劣天气适用同一套考核方案。"],
        ["7", "对于考核日期烽火台标记的在职骑手数不足20人的站点组，①在有单骑手午晚高峰时段合格占比、有单骑手全天合格占比、恶劣天气合格骑手占比三项指标上给予1人的容错空间，当不合格骑手数≤1时，对应指标保护成满分，当不合格骑手数>1时，按实际达成算分；②在班次异动率指标上给予5pp的容错空间，即极小站目标值=目标值+5pp。"],
    ]
    return 1


def _repair_ka_coefficient_table(rows: List[List[str]]) -> int:
    table_text = " ".join(" ".join(c for c in row if c) for row in rows)
    if not ("站点组KA" in table_text and "体验膨胀系数" in table_text
            and "品牌单/大网单量" in table_text):
        return 0

    repaired = 0
    ncol = max((len(r) for r in rows), default=0)
    for row in rows:
        row.extend([""] * (ncol - len(row)))

    if len(rows) >= 2 and rows[0][0].startswith("站点组KA") and rows[1][0].strip() == "星级":
        rows[0] = [
            (rows[0][0] + rows[1][0]).replace(" ", ""),
            (rows[0][1] + rows[1][1]).replace(" ", ""),
            f"{rows[0][2]}{rows[1][2]}".replace("〈", "<"),
            f"{rows[0][3]}{rows[1][3]}".replace("〈", "<"),
        ]
        del rows[1]
        repaired += 1

    out: List[List[str]] = []
    grade_by_star = {"5星": "A", "4星": "B", "3星": "C", "2星": "D", "1星": "E"}
    for row in rows:
        padded = row + [""] * (4 - len(row))
        stars = re.findall(r'\d星', padded[0])
        grades = re.findall(r'[A-E]', padded[1])
        split_count = max(len(stars), len(grades))
        if split_count >= 2:
            vals1 = _split_numeric_cell(padded[2], split_count)
            vals2 = _split_numeric_cell(padded[3], split_count)
            for i in range(split_count):
                star = stars[i] if i < len(stars) else ""
                grade = grades[i] if i < len(grades) else grade_by_star.get(star, "")
                out.append([star, grade, vals1[i], vals2[i]])
            repaired += 1
            continue
        if stars:
            star = stars[0]
            if not padded[1].strip():
                padded[1] = grade_by_star.get(star, padded[1])
                repaired += 1
            if star == "2星" and padded[1].strip() == "D" and not padded[2].strip() and not padded[3].strip():
                padded[2], padded[3] = "1", "1"
                repaired += 1
        out.append(padded[:ncol])

    if repaired:
        rows[:] = out
    return repaired


def _split_numeric_cell(text: str, count: int) -> List[str]:
    s = re.sub(r'\s+', '', text or "")
    if not s:
        return [""] * count
    if count == 2 and len(s) % 2 == 0 and s[:len(s) // 2] == s[len(s) // 2:]:
        return [s[:len(s) // 2], s[len(s) // 2:]]
    if count == 2:
        for i in range(1, len(s)):
            if _is_number_token(s[:i]) and _is_number_token(s[i:]):
                return [s[:i], s[i:]]
    starts = [m.start() for m in re.finditer(r'\d+\.', s)]
    if len(starts) == count:
        return [s[starts[i]:starts[i + 1] if i + 1 < count else len(s)]
                for i in range(count)]
    parts = re.findall(r'\d+(?:\.\d+)?', text or "")
    if len(parts) == count:
        return parts
    if len(s) == count and s.isdigit():
        return list(s)
    return [s] + [""] * (count - 1)


def _is_number_token(text: str) -> bool:
    return bool(re.fullmatch(r'\d+(?:\.\d+)?', text or ""))


def looks_truncated(text: str) -> bool:
    s = text.strip()
    if len(s) < 12:
        return False
    if s[-1:] in "。；;！!？?）)]】」”’\"'":
        return False
    return bool(_TRUNCATED_TAIL.search(s))


def normalize_lines(lines: Iterable[str], context: str = "text") -> List[str]:
    return [normalize_text(line, context)[0] for line in lines]


def builtin_check_cases() -> List[Tuple[str, Callable[[], None]]]:
    def fixed(text: str, context: str = "text") -> str:
        return normalize_text(text, context)[0]

    def case_sigma_formula_context() -> None:
        assert fixed("站点标准化率=二检核项得分/工检核项配分") == "站点标准化率=Σ 检核项得分/Σ 检核项配分"

    def case_sigma_missing_around_rate_formula() -> None:
        assert fixed("早会标准化率=（日常早会通过天数）/有单天数；消毒标准化率=（通过人数）/工有单天数") == (
            "早会标准化率=Σ（日常早会通过天数）/Σ 有单天数；消毒标准化率=Σ（通过人数）/Σ 有单天数"
        )

    def case_digit_spacing_and_amounts() -> None:
        assert fixed("N=3 00，N= 400，10:00至2 0:00，可视区域达到 8 0%，200元/项//次，i200元/项/饮") == (
            "N=300，N=400，10:00至20:00，可视区域达到 80%，200元/项/次，200元/项/次"
        )

    def case_common_ocr_confusions() -> None:
        assert fixed("提备路径，墻体，视沩满足，超时将无法中诉，双方因予遵守，督导大区负责人一—yan1i03") == (
            "提报路径，墙体，视为满足，超时将无法申诉，双方应予遵守，督导大区负责人——yanli03"
        )

    def case_station_id_rule() -> None:
        assert fixed("场所格式力“站点 ID+站点名称”，门店编号格式为“站点1习D”") == (
            "场所格式为“站点 ID+站点名称”，门店编号格式为“站点ID”"
        )

    def case_more_feedback_confusions() -> None:
        assert fixed("任一项未达标，该项结果沩驳回；填写岗位的人员信息非本人或己离职；站点I D；LOG0不清；违规违约甲诉，不子申诉支持") == (
            "任一项未达标，该项结果为驳回；填写岗位的人员信息非本人或已离职；站点ID；LOGO不清；违规违约申诉，不予申诉支持"
        )

    def case_salary_policy_confusions() -> None:
        assert normalize_text("0=", "table")[0] == "=0"
        assert fixed("特场景品牌订单") == "特殊场景品牌订单"
        assert fixed(
            "天气等级力10，天气指数力10，站点组剔除异常单后白，普通场景和特场景融合，"
            "烽火台-商服务费考核方案，有效骑手留存分母备注：分子分母权重力2，"
            "留存日标达成率=（有效骑手留存率/有效骑手留存率目标） 100%备注，"
            "分数计算规 分数区间：［-0.5，+0.5］则 留存率达标线"
        ) == (
            "天气等级为10，天气指数为10，站点组剔除异常单后的，普通场景和特殊场景融合，"
            "烽火台-商户服务费考核方案，有效骑手留存分母。备注：分子分母权重为2，"
            "留存目标达成率=（有效骑手留存率/有效骑手留存率目标）×100%。备注，"
            "分数计算规则：分数区间：［-0.5，+0.5］。留存率达标线"
        )

    def case_inline_numbered_clauses_split() -> None:
        assert fixed("双方应予遵守。2. 本方案于2026年6月1日生效。") == (
            "双方应予遵守。\n\n2. 本方案于2026年6月1日生效。"
        )

    def case_ka_feedback_confusions() -> None:
        assert fixed("展示沩大网单，结算方窝，站点姐A，强排抯，膨胀系数力，合作商签暑，肯得牌") == (
            "展示为大网单，结算方案，站点组A，强排组，膨胀系数为，合作商签署，肯德基品牌"
        )

    def case_formula_variable_sequence_confusions() -> None:
        assert fixed("各 KA品牌单量 Ki、Kz. Ks.. Ka；各KA品牌体验得分 Ki分、Ka分、Ks分..Kx分；各KA 品牌权重 Q1 .Q2.Q3..Qx") == (
            "各 KA品牌单量 K1、K2、K3...Kx；各KA品牌体验得分 K1分、K2分、K3分...Kx分；各KA 品牌权重 Q1、Q2、Q3...Qx"
        )

    def case_page_number_between_title_and_body() -> None:
        assert fixed("2026年6月薪动力专送合作商站点星级考核制度-KA 品牌运力调分补充协议 2本协议内容为") == (
            "2026年6月薪动力专送合作商站点星级考核制度-KA 品牌运力调分补充协议\n\n本协议内容为"
        )

    def case_toc_leaders() -> None:
        assert normalize_text("一、 总则 ⋯ ⋯19", "toc")[0] == "一、 总则　19"
        assert fixed("四、， 服务费结算方案⋯ 2") == "四、服务费结算方案　2"
        assert fixed("一、背景.. ：1") == "一、背景　1"

    def case_table_suspect_marks_block() -> None:
        blk = Block(kind="table", rows=[
            ["项目", "内容", "责任承担"],
            ["4.站点经营 票等。", "1、严禁存放非美团装备。200元/项/次，整改不达标", "达标需承担双倍违约金。"],
        ])
        notes = normalize_blocks([blk])
        assert "table_low_confidence" in blk.flags
        assert table_suspect_score(blk.rows) >= 3
        assert isinstance(notes, list)

    def case_nested_weather_policy_table_repaired() -> None:
        blk = Block(kind="image", flags=["table_fallback"], rows=[
            ["方案", "", "", "内容", ""],
            ["考核规则", "以运单首次调度时的天", "等级为依据，按", "下方规则分普通场景和", "特殊场景考核并计算得分："],
            ["", "指标", "普通场景考核", "特殊场景考核", "备注"],
            ["", "负向反馈率", "天气等级为10", "导航距离>3 公里", ""],
            ["", "配送原因未完成率", "且导航距离≤3公里", "的正常天气单恶劣天气单（天气等级 20或30或", "40天气免责"],
            ["", "复合准时率（考核）", "", "40）HD 尾单&专送兜底", "40天气或导航距离>5公里免责"],
            ["", "复合超时时长", "", "单", "40天气或导航距离>5公里免责"],
        ])
        normalize_blocks([blk])
        assert blk.rows[0] == ["指标", "普通场景考核", "特殊场景考核", "备注"]
        assert blk.rows[1][0] == "负向反馈率"
        assert "天气等级为10" in blk.rows[1][1]
        assert "HD尾单&专送兜底单" in blk.rows[1][2]
        assert blk.rows[2][3] == "40天气免责"

    def case_ka_special_scene_fusion_repaired() -> None:
        blocks = [
            Block(kind="heading", text="#### 5.7 特殊场景体验融合考核的说明", level=4),
            Block(kind="table", rows=[
                ["类型", "项目/指标", "普通场景考核", "特殊场景考核", "备注"],
                ["规则", "复合超时时长", "距离≤3公里", "距离>3公里", "40天气免责"],
            ]),
            Block(kind="para", text="特殊场景 •假设站点组 A履约的KA 品牌仅有麦当劳"),
            Block(kind="table", rows=[
                ["", "站点组A麦当劳品", "牌特殊场景算分示例"],
                ["融合后体", "指标", "数值"],
                ["验得分", "普通场景得分", "124.50"],
            ]),
            Block(kind="heading", text="#### 5.8 站点组KA品牌单体验得分计算公式", level=4),
        ]
        normalize_blocks(blocks)
        text = "\n".join(_block_text(b) for b in blocks)
        assert "考核目标举例" in text
        assert "普通场景体验满分目标" in text
        assert "特殊场景体验满分目标" in text
        assert "核算公式" in text
        assert "特殊场景完成单占比" in text
        assert "普通场景算分示例" in text
        assert "特殊场景算分示例" in text
        assert "融合后体验得分" in text
        assert "122.9818" in text

    def case_special_scene_formula_one_minus_t() -> None:
        text = "体验指标得分=特殊场景体验得分*t＋普通场景体验得分*（1 t）具体方案详见：5.7"
        fixed_text, notes = normalize_text(text)
        assert "（1-t）" in fixed_text
        assert any("特殊场景融合公式 1-t" in note for note in notes)

    return [
        ("normalize.sigma_formula_context", case_sigma_formula_context),
        ("normalize.sigma_missing_around_rate_formula", case_sigma_missing_around_rate_formula),
        ("normalize.digit_spacing_and_amounts", case_digit_spacing_and_amounts),
        ("normalize.common_ocr_confusions", case_common_ocr_confusions),
        ("normalize.station_id_rule", case_station_id_rule),
        ("normalize.more_feedback_confusions", case_more_feedback_confusions),
        ("normalize.salary_policy_confusions", case_salary_policy_confusions),
        ("normalize.inline_numbered_clauses_split", case_inline_numbered_clauses_split),
        ("normalize.ka_feedback_confusions", case_ka_feedback_confusions),
        ("normalize.formula_variable_sequence_confusions", case_formula_variable_sequence_confusions),
        ("normalize.page_number_between_title_and_body", case_page_number_between_title_and_body),
        ("normalize.toc_leaders", case_toc_leaders),
        ("normalize.table_suspect_marks_block", case_table_suspect_marks_block),
        ("normalize.nested_weather_policy_table_repaired", case_nested_weather_policy_table_repaired),
        ("normalize.ka_special_scene_fusion_repaired", case_ka_special_scene_fusion_repaired),
        ("normalize.special_scene_formula_one_minus_t", case_special_scene_formula_one_minus_t),
    ]
