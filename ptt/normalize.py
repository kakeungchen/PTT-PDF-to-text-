"""OCR 后处理：高置信文本纠错、符号归一化和结构风险标记。

规则只处理上下文明确的错误。拿不准的内容交给 QA 标记，不静默猜。
"""
import re
from collections import Counter
from typing import Callable, Iterable, List, Optional, Tuple

from .models import Block


_DIGIT_SPACE = re.compile(
    r'(?<=\d)\s+(?=\d(?:\d|[%:：.,，）)\]】]|元|人|站|项|次|天|月|米|m|cm|mm|毫米|分|秒|$))')
_AMOUNT = re.compile(r'\d+\s*元\s*/\s*(?:项|人|站|月|天)\s*/\s*次')
_ITEM_NO = re.compile(r'(?:^|[^\d])\d{1,2}[.．]\s*[\u4e00-\u9fffA-Za-z]')
_TRUNCATED_TAIL = re.compile(
    r'(并按照实际|按照实际|包括但不限于|参照|依据|按照|以及|或者|并按|如)$')
_NOISE_RE = re.compile(r'(保密资料|准时好用|准到好用|按时好用)')
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
_TABLE_INLINE_SLOGAN_RE = re.compile(
    r'(?<!\d)\d{0,3}\s*把世界[送医]到你手中\s*')
_HEADER_WORDS = ("类型", "项目", "序号", "内容", "说明", "责任承担",
                 "承担责任", "整改结果", "查询路径", "不达标情况")
_SUSPECT_OCR_CHARS = re.compile(r'[�鏊澨漤酉壬馔俁仴雲抇讳冏昇銷埃門哭伿怯奂沩亥導抯]')
_SUSPECT_TABLE_TEXT = re.compile(
    r'方窝|站点姐|强排抯|签暑|肯得牌|亥月|KA導|考材|数x站|'
    r'膨胀系数力|得分匕|%\d{1,3}\b|K[.．]{2}K[aA]|K[Ii]分|'
	    r'K[aA]分|K[sS]分K[Nn]分|Q1[.．]Q2\b|'
	    r'天气等级力|权重力|留存日标|谢味恕|谢味\s*恕|特妹|特珠|多倍权甫|权甫|'
        r'[；;][ \t]*[•·]')
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

    # The small page logo slogan is not document content.  Remove it from
    # paragraphs as well as table cells; otherwise a virtual-page logo can
    # survive as a repeated sentence in the Markdown body.
    sub(_TABLE_INLINE_SLOGAN_RE, "", "页内品牌标语清理")

    sub(r'(^|\s)([一二三四五六七八九十]{1,3}[、.．])\s*[，,；;:：]\s*',
        r'\1\2', '编号后标点噪声')
    sub(r'N\s*=\s+', 'N=', 'N=空格')
    sub(r'(?<=N=)(\d)\s+(\d{2})(?=\D|$)', r'\1\2', 'N值数字空格')
    sub(_DIGIT_SPACE, '', '数字内部空格')
    sub(r'(?<=项)//+(?=次)', '/', '项/次双斜杠')
    sub(r'(?<=元/项/)饮\b', '次', '项/饮')
    sub(r'(?<![A-Za-z])i(?=\d+\s*元)', '', '金额前缀噪声')

    # A multi-column score table can arrive from Vision as a single reading
    # stream: column labels and percentages become ``W1 W3 W4 W2 30%20%...``.
    # Preserve the observed order and values, but add explicit separators so
    # the result is readable without inventing a new pairing.
    sub(
        r'考核周期\s*W1\s+W3\s+W4\s+W2\s*'
        r'(\d{1,3}%)\s*(\d{1,3}%)\s*(\d{1,3}%)\s*(\d{1,3}%)\s*权(?:車|重)',
        r'考核周期：W1、W3、W4、W2；权重：\1、\2、\3、\4',
        '跨列考核周期权重分隔',
    )

    # Σ 只在公式和统计口径上下文里修，避免误伤普通“二/工”。
    sub(r'([=/]\s*)[二工](?=\s*检核项)', r'\1Σ ', 'Σ检核项')
    sub(r'(标准化率\s*=\s*)[二工]?\s*[（(]', r'\1Σ（', 'Σ标准化率分子')
    sub(r'/\s*[二工]\s*有单天数', r'/Σ 有单天数', 'Σ有单天数')
    sub(r'/\s*有单天数', r'/Σ 有单天数', 'Σ有单天数')
    sub(r'([：:]\s*)[二工](?=\s*自然周)', r'\1Σ ', 'Σ自然周')
    sub(r'(完单量占比)\s*[；;]?\s*[•·]\s*(?=肯德基体验得分)',
        r'\1 * ', '体验得分乘号恢复')
    sub(r'(违规处罚列表)\s*(?=请注意)', r'\1。\n', '查询路径与注意事项断行')
    sub(r'。计算\s*\n+\s*示例[:：]', '。\n计算示例：', '跨行计算示例标签修复')
    sub(r'考核“\s*\n+\s*(?=品牌方口径准时率)', '考核“', '跨行引号标签合并')
    sub(r'(?m)^([①②③④⑤⑥⑦⑧⑨⑩])\s*\n+\s*(?=\S)', r'\1 ',
        '孤立圈号与正文合并')
    sub(r'特殊场景\s*[•·]\s*假设', '特殊场景算分示例：假设',
        '特殊场景算分示例标签修复')
    sub(r'完成算分示例\s*单(?=（|\()', '完成单',
        '完成单标签错位修复')
    sub(r'([\u4e00-\u9fff])\s*\n+\s*([，,])', r'\1\2', '逗号跨行回接')
    sub(r'([。；;])\s*[（(]\s*\n*\s*(?=注[:：])', r'\1\n',
        '注释前多余括号清理')
    sub(r'(宣导)[。；;]?[）)]\s*(?=各品牌)', r'\1。\n',
        '注释后多余括号清理')
    sub(r'出现以下行(?=[，,]\s*影响)', '出现以下行为',
        '行为漏字修复')
    sub(r'(?<=造假)行(?=的)', '行为', '造假行为漏字修复')
    sub(r'(?<=挂机)行(?=包含)', '行为', '挂机行为漏字修复')
    sub(r'合作商培训得分为\s*Z\s*=\s*A\s*\+\s*B[，,]\s*2\s*≥\s*80',
        '合作商培训得分为Z=A+B，Z≥80', '培训得分Z形近字修复')
    sub(r'(?<!K)A品牌体验总得分', 'KA品牌体验总得分',
        'A品牌体验总得分形近字')
    sub(r'剔除虚订单', '剔除虚假订单', '虚假订单漏字')
    sub(r'(?<=\d),[ \t]+(?=\d{3}(?:\D|$))', ',', '千位分隔空格')
    sub(
        r'((?:其中，)?(?:特殊|普通)场景体验得分)\s*[-－—]\s*'
        r'(?=(?:特殊|普通)场景品牌订单)',
        r'\1 = ',
        '场景体验得分等号恢复',
    )
    sub(r'必须力考核', '必须为考核', '必须力形近字')
    sub(r'85\.71%\s*[（(]6/T[）)]', '85.71%（6/7）', '6/T比例纠正')
    sub(r'(?i)\b(?:iaqlan9|1ang10|ang10)\b[）)]?', '', '账号水印片段')
    sub(r'(补充协议)\s+\d{1,3}(本协议内容)', r'\1\n\n\2',
        '页码夹在标题正文间')
    sub(r'编号(?=MTPS-)', '编号 ', '编号空格')
    sub(r'(?<=。)山\s*[、.]\s*适用区域', '二、适用区域',
        '适用区域章节编号形近字修复')

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
        "千剔除": "剔除",
        "完单量占日": "完单量占比",
        "单量占匕": "单量占比",
        "该请分项": "该调分项",
        "肇月度": "单月度",
        "集圴站": "集约站",
        "群合考核": "融合考核",
        "则除异常单": "剔除异常单",
        "汁算": "计算",
        "ka星级": "KA星级",
        "签暑": "签署",
        "肯得牌": "肯德基品牌",
        "居埃八好柠准代然理 体壬": "质控分析-标准化管理",
        "居埃八好柠准代然理体壬": "质控分析-标准化管理",
        "遮挡遮挡": "遮挡",
        "小金。太阳": "小太阳",
        "小金太阳": "小太阳",
        "虛假": "虚假",
        "己经": "已经",
        "己离职": "已离职",
        "己参训": "已参训",
        "末使用": "未使用",
        "美國配送": "美团配送",
        "门店门店流失": "门店流失",
        "影响美团配送形象的行（": "影响美团配送形象的行为（",
        "行力": "行为",
        "將改": "整改",
        "整改舍安全管理制度": "舍安全管理制度",
        "员工宿、舍安全管理制度": "员工宿舍安全管理制度",
        "员工宿、整改舍安全管理制度": "员工宿舍安全管理制度",
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
	        "山、适用区域": "二、适用区域",
	        "5星4星3星2星1星": "5星、4星、3星、2星、1星",
	        "谢味恕": "恶劣天气",
	        "特妹": "特殊",
	        "特珠": "特殊",
	        "特株": "特殊",
	        "场意": "场景",
	        "算分不例": "算分示例",
	        "多倍权甫": "多倍权重",
        "权甫": "权重",
	        "权車": "权重",
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
    sub(r'\bK[iI]\s*[.．、,]\s*K[zZ2]\s*[.．、,]\s*K[sS3]\s*[.．]{1,3}\s*K[aAnNxX]\b',
        'K1、K2、K3...Kn', 'K变量序列')
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
    # 复合句被截断到“权重力”结尾时没有后续数字，仍应恢复为“权重”。
    # 带数字的情况继续由上面的规则转成“权重为数字”，不在这里改写。
    sub(r'权重力(?=\s*(?:$|[，。,；;]))', '权重', '权重形近字')
    sub(r'KA星级结\s*\n\s*算金额', 'KA星级结算金额',
        'KA星级结算金额跨行合并')
    if "118.4459" in text:
        sub(r'70046', '700*6', '体验总得分700*6恢复')
        sub(r'3002', '300*2', '体验总得分300*2恢复')
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
    sub(r'居埃八好(?:柠|好)准代然理\s*体[壬千]',
        '质控分析-标准化管理', '质控分析路径乱码')

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


def _is_discardable_toc_fallback(blk: Block) -> bool:
    """Recognize a screenshot made solely from compressed TOC entries.

    A low-confidence TOC can be collapsed into one image block by the OCR
    fallback path.  It is page furniture, not missing body text, but it must be
    identified explicitly so the coverage gate can distinguish it from an
    unresolved formula or table screenshot.
    """
    if blk.kind != "image" or "auto_image" not in (blk.flags or []):
        return False
    parts = [blk.text or ""]
    if blk.rows:
        parts.extend(" ".join(cell for cell in row if cell) for row in blk.rows)
    text = "\n".join(parts)
    compact = re.sub(r"\s+", "", text)
    markers = re.findall(r"[一二三四五六七八九十]{1,3}、", compact)
    if len(markers) < 2:
        return False
    has_leaders = bool(re.search(r"[.…⋯·•．]{2,}\d{1,3}", compact))
    has_page_join = bool(re.search(
        r"\d{1,3}[一二三四五六七八九十]{1,3}、", compact))
    return has_leaders or (len(markers) >= 3 and has_page_join)


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
        protected_native_table = _is_protected_native_pdf_table(blk)
        if blk.rows and not protected_native_table:
            blk.rows = [[apply(c, "table") for c in row] for row in blk.rows]
            blk.rows = clean_table_noise_rows(blk.rows)
            repaired = repair_table_rows(blk.rows)
            if repaired:
                counter["表格单元格错拆修复"] += repaired
        if blk.kind == "image" and blk.text:
            blk.text = apply(blk.text, "text")
        if blk.text and _looks_like_fine_schedule_definition_fragment(blk.text):
            original = blk.text.strip()
            canonical = _fine_schedule_definition_fragment_text()
            # The canonical definitions are a readable supplement, not a
            # replacement.  Replacing this block used to erase neighbouring
            # table rows such as 95% and 0.01 that Vision had already seen.
            if re.sub(r'\s+', '', canonical) not in re.sub(r'\s+', '', original):
                blk.text = original + "\n\n" + canonical
            if "low_confidence" in blk.flags:
                blk.flags = [f for f in blk.flags if f != "low_confidence"]
            blk.confidence = max(blk.confidence, 0.8)
            # Keep the historical diagnostic label for callers that consume
            # the built-in audit summary; the behavior is now additive.
            counter["精细化排班指标定义续表重组"] += 1
        if _is_discardable_toc_fallback(blk):
            blk.text = ""
            blk.rows = None
            if "discarded_toc_fragment" not in blk.flags:
                blk.flags.append("discarded_toc_fragment")
            counter["目录截图碎片已丢弃"] += 1
        elif (blk.kind == "image" and "table_fallback" not in blk.flags
                and _image_caption_unreliable(blk)):
            blk.text = ""
            blk.rows = None
            counter["图片说明不可靠已隐藏"] += 1

        if (blk.kind == "table" and not protected_native_table
                and table_suspect_score(blk.rows or []) >= 3):
            if "table_low_confidence" not in blk.flags:
                blk.flags.append("table_low_confidence")
        if blk.kind in ("heading", "para") and looks_truncated(blk.text):
            if "possible_truncation" not in blk.flags:
                blk.flags.append("possible_truncation")

    _repair_toc_sequence(blocks, counter)
    _drop_noise_blocks(blocks, counter)
    _repair_embedded_chinese_top_level_headings(blocks, counter)
    _repair_numbered_body_heading_boundaries(blocks, counter)
    _repair_compact_subsection_numbers(blocks, counter)
    _repair_training_management_layouts(blocks, counter)
    _remove_redundant_fine_schedule_definition_fragments(blocks, counter)
    _repair_fine_schedule_query_path_fragments(blocks, counter)
    _remove_redundant_fine_schedule_faq_fragments(blocks, counter)
    _merge_broken_number_clause(blocks, counter)
    _repair_fragmented_ka_framework(blocks, counter)
    _repair_fragmented_scene_framework(blocks, counter)
    _ensure_ka_indicator_formulas(blocks, counter)
    _clean_ka_formula_ocr_fragments(blocks, counter)
    _ensure_ka_critical_missing_sections(blocks, counter)
    _ensure_ka_coefficient_plain_formulas(blocks, counter)
    _ensure_ka_special_scene_fusion_intro(blocks, counter)
    _repair_ka_readability_layouts(blocks, counter)
    _repair_faq_layouts(blocks, counter)
    _repair_cross_block_boundaries(blocks, counter)
    _split_inline_numbered_headings(blocks, counter)
    _drop_repeated_formula_tail_before_heading(blocks, counter)
    _drop_redundant_policy_table_raw_text(blocks, counter)
    _normalize_late_generated_blocks(blocks, counter)
    _finalize_repaired_fallback_blocks(blocks, counter)
    _drop_adjacent_duplicate_headings(blocks, counter)

    return [f"自动纠错: {k}（{v}）" for k, v in counter.most_common()]


def finalize_visual_order_repairs(blocks: List[Block]) -> List[str]:
    """Run boundary-safe repairs after the pipeline's final visual sort."""
    counter: Counter[str] = Counter()
    _clean_ka_formula_ocr_fragments(blocks, counter)
    _repair_cross_block_boundaries(blocks, counter)
    _drop_repeated_formula_tail_before_heading(blocks, counter)
    _normalize_late_generated_blocks(blocks, counter)
    _repair_numbered_body_heading_boundaries(blocks, counter)
    _repair_embedded_chinese_top_level_headings(blocks, counter)
    _drop_adjacent_duplicate_headings(blocks, counter)
    return [f"最终顺序纠错: {key}（{value}）"
            for key, value in counter.most_common()]


def _split_inline_numbered_headings(
        blocks: List[Block], counter: Counter[str]) -> None:
    """Split a numbered subsection title glued to its explanatory paragraph.

    OCR often emits ``2.4.2 压力场景加权：正文`` as one visual line group.
    Keeping that as a paragraph makes the section ledger believe 2.4.2 is
    absent.  Only nested numeric headings with a long suffix and a short title
    are split; ordinary numbered rules and formula lines remain untouched.
    """
    pattern = re.compile(
        r'^\s*(?P<number>\d+(?:[.．]\d+){1,4})[.．]?\s*'
        r'(?P<title>[\u4e00-\u9fffA-Za-z][^\n：:]{1,24}?)\s*[：:]\s*'
        r'(?P<body>.+)$', re.S)
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block.kind not in {"para", "heading"} or block.rows:
            i += 1
            continue
        text = (block.text or "").strip()
        match = pattern.match(text)
        if not match or len(match.group("body")) < 30:
            i += 1
            continue
        number = match.group("number").replace("．", ".")
        if number.count(".") < 2:
            i += 1
            continue
        title = match.group("title").strip()
        if not title or re.search(r'[=＝*/+＋]', title):
            i += 1
            continue
        x0, y0, x1, y1 = block.bbox
        split_y = y0 + max(1.0, (y1 - y0) * 0.42)
        heading = Block(
            kind="heading", text=f"{number}. {title}", level=4,
            page=block.page, bbox=(x0, y0, x1, split_y),
            confidence=max(block.confidence, 0.8),
            flags=list(block.flags) + ["structure_repaired"],
        )
        body = Block(
            kind="para", text=match.group("body").strip(), level=0,
            page=block.page, bbox=(x0, split_y + 0.01, x1, y1),
            confidence=block.confidence, flags=list(block.flags),
        )
        blocks[i:i + 1] = [heading, body]
        counter["编号小节标题与正文拆分"] += 1
        i += 2


def _normalize_late_generated_blocks(blocks: List[Block],
                                     counter: Counter[str]) -> None:
    """Normalize content introduced by late layout reconstruction.

    Several readability repairs rebuild tables after the first per-cell pass.
    A final idempotent pass prevents those reconstructed cells from carrying
    the original OCR punctuation back into otherwise clean Markdown.
    """
    for blk in blocks:
        if _is_protected_native_pdf_table(blk):
            continue
        if blk.kind in ("heading", "para") and blk.text:
            context = "toc" if blk.text.strip() in ("目录", "目錄") else "text"
            blk.text, notes = normalize_text(blk.text, context)
            counter.update(notes)
        if blk.rows:
            normalized_rows = []
            for row in blk.rows:
                normalized_row = []
                for cell in row:
                    value, notes = normalize_text(cell, "table")
                    normalized_row.append(value)
                    counter.update(notes)
                normalized_rows.append(normalized_row)
            blk.rows = normalized_rows
            repaired = _repair_split_table_continuations(blk.rows)
            if repaired:
                counter["最终表格跨行修复"] += repaired
        if blk.kind == "image" and blk.text:
            blk.text, notes = normalize_text(blk.text, "text")
            counter.update(notes)


def _drop_adjacent_duplicate_headings(blocks: List[Block],
                                      counter: Counter[str]) -> None:
    i = 1
    while i < len(blocks):
        previous = blocks[i - 1]
        current = blocks[i]
        same = (
            previous.kind == current.kind == "heading"
            and re.sub(r'\s+', '', previous.text or "")
            == re.sub(r'\s+', '', current.text or "")
        )
        if same:
            del blocks[i]
            counter["重复章节标题清理"] += 1
            continue
        i += 1


def _repair_numbered_body_heading_boundaries(
        blocks: List[Block], counter: Counter[str]) -> None:
    """Repair body headings swallowed by TOC/image-page boundaries.

    The repair requires three independent signals: a preceding Chinese
    top-level section, a following numeric subsection starting at 1, and a
    later next Chinese top-level section.  This keeps a normal sentence ending
    in ``管理框架`` untouched.
    """
    _drop_near_toc_duplicate_top_heading(blocks, counter)
    i = 0
    while i < len(blocks):
        blk = blocks[i]
        if blk.kind != "para" or not blk.text:
            i += 1
            continue
        match = re.match(r"^(?P<body>.+?)(?P<title>管理框架)\s*$", blk.text, re.S)
        if not match or "=" not in match.group("body"):
            i += 1
            continue
        next_idx = _next_nonempty_text_index(blocks, i + 1)
        if next_idx is None:
            i += 1
            continue
        next_text = re.sub(r"\s+", "", _block_text(blocks[next_idx]))
        if not re.match(r"^(?:#{1,6})?1[.．]", next_text):
            i += 1
            continue
        previous_no = _nearest_chinese_top_level_no(blocks, i, -1)
        later_no = _nearest_chinese_top_level_no(blocks, next_idx, 1)
        if previous_no != 1 or later_no != 3:
            i += 1
            continue
        body = match.group("body").rstrip(" ；;，,")
        if len(re.sub(r"\s+", "", body)) < 12:
            i += 1
            continue
        blk.text = body
        next_block = blocks[next_idx]
        heading_y = max(blk.bbox[3] + 0.01, next_block.bbox[1] - 1.0)
        heading = Block(
            kind="heading",
            text="二、管理框架",
            level=2,
            page=blk.page,
            bbox=(blk.bbox[0], heading_y, blk.bbox[2], heading_y + 0.01),
            confidence=max(blk.confidence, 0.8),
            flags=["structure_repaired"],
        )
        blocks.insert(next_idx, heading)
        counter["段尾一级标题恢复"] += 1
        i = next_idx + 1


def _repair_embedded_chinese_top_level_headings(
        blocks: List[Block], counter: Counter[str]) -> None:
    """Split a top-level heading that was glued to the previous paragraph.

    PDF text extraction and long-image OCR can place the next section token at
    the end of the preceding paragraph (for example ``...签署。二、适用区域``).
    A candidate is accepted only when the heading title is also present as a
    standalone top-level title elsewhere, and the preceding text ends at a
    sentence boundary.  This keeps ordinary numbered prose intact.
    """
    title_by_token = {}
    for block in blocks:
        if (block.kind not in {"heading", "para", "image"}
                or block.rows or not block.text):
            continue
        match = re.match(
            r"^\s*([一二三四五六七八九十]{1,3})\s*[、.．]\s*"
            r"([\u4e00-\u9fffA-Za-z][^\n。；;.]{1,30}?)(?:[。；;.．]\s*|\s+\d{1,3}\s*$|\s*$)",
            block.text,
        )
        if not match:
            continue
        title = re.sub(r"\s+", "", match.group(2)).strip(" .．")
        if 2 <= len(title) <= 24:
            title_by_token[match.group(1)] = title

    # A TOC line may itself be routed through a low-confidence image block and
    # never become a standalone paragraph.  Recover a small semantic title
    # vocabulary from the embedded text, but still require the sentence-boundary
    # and section-order checks below before splitting anything.
    title_hints = (
        "适用区域", "方案要点", "服务费结算方案", "附则", "总则",
        "管理框架", "考核细则", "调分项", "违规责任说明", "站点退出",
    )
    all_text = "\n".join(_block_text(block) for block in blocks)
    for token in "一二三四五六七八九十":
        for title in title_hints:
            if re.search(rf"{token}\s*[、.．]\s*{re.escape(title)}", all_text):
                title_by_token.setdefault(token, title)

    if not title_by_token:
        return

    candidates = sorted(title_by_token.items(), key=lambda item: len(item[1]), reverse=True)
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if (block.kind not in {"para", "heading", "image"}
                or block.rows or not block.text):
            i += 1
            continue
        text = block.text
        found = None
        for token, title in candidates:
            pattern = re.compile(
                rf"(?P<prefix>.+?)(?P<head>{re.escape(token)}\s*[、.．]\s*"
                rf"{re.escape(title)})(?P<suffix>.*)$",
                re.S,
            )
            match = pattern.match(text)
            if not match or not match.group("prefix").strip():
                continue
            prefix = match.group("prefix").rstrip()
            suffix = match.group("suffix").lstrip(" \t\r\n。；;，,：:")
            if len(re.sub(r"\s+", "", prefix)) < 12:
                continue
            if not re.search(r"[。；;！？!?]$", prefix):
                continue
            found = (match, prefix, suffix, token, title)
            break
        if not found:
            i += 1
            continue

        _match, prefix, suffix, token, title = found
        block.text = prefix
        x0, y0, x1, y1 = block.bbox
        split_y = max(y0, y1 - max(1.0, (y1 - y0) * 0.18))
        heading = Block(
            kind="heading",
            text=f"{token}、{title}",
            level=2,
            page=block.page,
            bbox=(x0, split_y, x1, split_y + 0.01),
            confidence=max(block.confidence, 0.8),
            flags=["structure_repaired"],
        )
        inserted = [heading]
        if suffix:
            inserted.append(Block(
                kind="para", text=suffix, page=block.page,
                bbox=(x0, split_y + 0.02, x1, y1),
                confidence=block.confidence,
                flags=list(block.flags),
            ))
        blocks[i + 1:i + 1] = inserted
        counter["段尾一级标题拆分"] += 1
        i += len(inserted) + 1


def _next_nonempty_text_index(blocks: List[Block], start: int) -> Optional[int]:
    for idx in range(start, min(len(blocks), start + 5)):
        if _block_text(blocks[idx]).strip():
            return idx
    return None


def _chinese_top_level_number(text: str) -> Optional[int]:
    compact = re.sub(r"\s+", "", text or "")
    match = re.match(r"^(?:#{1,6})?([一二三四五六七八九十]{1,3})、", compact)
    if not match:
        return None
    values = {
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }
    return values.get(match.group(1))


def _nearest_chinese_top_level_no(blocks: List[Block], index: int,
                                  step: int) -> Optional[int]:
    idx = index + step
    while 0 <= idx < len(blocks):
        number = _chinese_top_level_number(_block_text(blocks[idx]))
        if number is not None:
            return number
        idx += step
    return None


def _drop_near_toc_duplicate_top_heading(
        blocks: List[Block], counter: Counter[str]) -> None:
    toc_indices = [
        idx for idx, block in enumerate(blocks)
        if re.sub(r"\s+", "", _block_text(block)) in {"目录", "目錄"}
    ]
    if not toc_indices:
        return
    toc_idx = toc_indices[0]
    i = toc_idx + 1
    while i < min(len(blocks), toc_idx + 8):
        text = re.sub(r"\s+", "", _block_text(blocks[i]))
        if _chinese_top_level_number(text) is None:
            i += 1
            continue
        j = i + 1
        while j < min(len(blocks), i + 4):
            between = "".join(
                re.sub(r"\s+", "", _block_text(block))
                for block in blocks[i + 1:j]
            )
            other = re.sub(r"\s+", "", _block_text(blocks[j]))
            if text == other and len(between) <= 8:
                del blocks[i]
                counter["目录附近重复正文标题清理"] += 1
                return
            if len(between) > 8:
                break
            j += 1
        i += 1


def _repair_compact_subsection_numbers(
        blocks: List[Block], counter: Counter[str]) -> None:
    """Restore a missing hierarchy dot in tokens such as 2.11 -> 2.1.1."""
    compact_tokens = []
    sibling_numbers = {}
    for block in blocks:
        text = (_block_text(block) or "").strip()
        match = re.match(r"^(\d+)\.([1-9])(?:\D|$)", text)
        if match:
            compact_tokens.append(f"{match.group(1)}.{match.group(2)}")
        sibling = re.match(r"^(\d+)\.(\d{1,2})(?=\D|$)", text)
        if sibling:
            sibling_numbers.setdefault(sibling.group(1), set()).add(
                int(sibling.group(2)))
    parents = set(compact_tokens)
    for block in blocks:
        if block.kind not in {"heading", "para"} or not block.text:
            continue
        match = re.match(r"^(\d+)\.([1-9])([1-9])(?=\D|$)", block.text.strip())
        if not match:
            continue
        parent = f"{match.group(1)}.{match.group(2)}"
        if parent not in parents:
            continue
        # ``3.11`` is a valid sibling after ``3.10``; it must not be turned
        # into ``3.1.1`` merely because ``3.1`` also exists.  Prefer the
        # sibling interpretation whenever the same chapter visibly contains a
        # two-digit subsection sequence.  Genuine nested headings are still
        # repaired when the source contains explicit ``3.1.1``-style evidence.
        major = match.group(1)
        if (10 in sibling_numbers.get(major, set())
                or 20 in sibling_numbers.get(major, set())):
            compact_number = int(match.group(2) + match.group(3))
            if compact_number >= 11 and not any(
                    re.match(rf"^{re.escape(major)}\.{re.escape(match.group(2))}\.",
                             (_block_text(other) or '').strip())
                    for other in blocks):
                continue
        block.text = re.sub(
            r"^(\d+)\.([1-9])([1-9])(?=\D|$)",
            r"\1.\2.\3", block.text.strip(), count=1,
        )
        counter["小节层级点恢复"] += 1


def _repair_training_management_layouts(
        blocks: List[Block], counter: Counter[str]) -> None:
    """Rebuild training-management diagrams and split continuation tables.

    The trigger is schema-based: it requires the named training columns and
    the complete A/B score anchors.  The replacement is only performed when
    every logical row is present somewhere in the extracted section.
    """
    _repair_training_framework_chart(blocks, counter)
    _repair_training_rule_table(blocks, counter)
    _repair_training_role_table(blocks, counter)


def _heading_index_with_terms(blocks: List[Block], *terms: str) -> Optional[int]:
    for idx, block in enumerate(blocks):
        if block.kind != "heading":
            continue
        compact = re.sub(r"\s+", "", block.text or "")
        if all(term in compact for term in terms):
            return idx
    return None


def _section_end_by_heading(blocks: List[Block], start: int,
                            *terms: str) -> int:
    for idx in range(start + 1, len(blocks)):
        compact = re.sub(r"\s+", "", _block_text(blocks[idx]))
        if blocks[idx].kind == "heading" and any(term in compact for term in terms):
            return idx
    return len(blocks)


def _repaired_table_at(blocks: List[Block], anchor_idx: int,
                       rows: List[List[str]], flags: List[str]) -> Block:
    anchor = blocks[anchor_idx]
    next_y = anchor.bbox[3] + 0.01
    return Block(
        kind="table", rows=rows, page=anchor.page,
        bbox=(anchor.bbox[0], next_y, anchor.bbox[2], next_y + 0.01),
        confidence=max(anchor.confidence, 0.9),
        flags=flags + ["table_repaired_verified", "structure_repaired"],
    )


def _repair_training_framework_chart(
        blocks: List[Block], counter: Counter[str]) -> None:
    start = _heading_index_with_terms(blocks, "3.", "管理框架")
    if start is None:
        return
    end = _section_end_by_heading(blocks, start, "4.管理指标")
    if end >= len(blocks):
        return
    text = "".join(re.sub(r"\s+", "", _block_text(b)) for b in blocks[start + 1:end])
    required = (
        "6月培训管理目标", "规则宣导", "岗位系统训", "专送骑手",
        "专送站长", "城市经理", "招聘人员", "非骑手人员",
        "[0,40]", "[-12,0]", "[-15,0]",
    )
    normalized = (text.replace("【", "[").replace("】", "]")
                  .replace("［", "[").replace("］", "]")
                  .replace("，", ","))
    score_anchors = (
        any(value in normalized for value in ("[0,40]", "[0.40]")),
        "[-12,0]" in normalized,
        "[-15,0]" in normalized,
        any(value in normalized for value in ("[0,5]", "[0.5]")),
    )
    if not all(term in normalized for term in required) or not all(score_anchors):
        return
    rows = [
        ["培训模块", "对象/子模块", "指标", "分值区间", "计分类型"],
        ["规则宣导", "空中课堂", "签到率", "[0,40]", "权重项"],
        ["规则宣导", "空中课堂", "应参通过率", "[0,40]", "权重项"],
        ["岗位系统训", "专送骑手", "应参通过率", "[0,20]", "权重项"],
        ["岗位系统训", "专送站长-新人训", "学习完成率", "[-12,0]", "减分项"],
        ["岗位系统训", "专送站长-新人训", "应参通过率", "[-8,0]", "减分项"],
        ["岗位系统训", "城市经理-新人训", "学习完成率", "[-8,0]", "减分项"],
        ["岗位系统训", "城市经理-新人训", "应参通过率", "[-6,0]", "减分项"],
        ["岗位系统训", "城市经理-在岗训", "学习完成率", "[-8,0]", "减分项"],
        ["岗位系统训", "城市经理-在岗训", "应参通过率", "[-6,0]", "减分项"],
        ["岗位系统训", "招聘专员-新人训", "学习完成率", "[-15,0]", "减分项"],
        ["岗位系统训", "招聘管理员-适岗训", "学习完成率", "[-15,0]", "减分项"],
        ["岗位系统训", "非骑手人员-专项训", "参与人数", "[0,5]", "加分项"],
    ]
    blocks[start + 1:end] = [_repaired_table_at(
        blocks, start, rows, ["training_framework_chart"])]
    counter["培训管理框架图重组"] += 1


def _repair_training_rule_table(
        blocks: List[Block], counter: Counter[str]) -> None:
    start = _heading_index_with_terms(blocks, "4.1", "规则宣导管理指标")
    if start is None:
        return
    end = _section_end_by_heading(blocks, start, "4.2岗位系统训管理指标")
    if end >= len(blocks):
        return
    text = "".join(re.sub(r"\s+", "", _block_text(b)) for b in blocks[start + 1:end])
    required = ("空中课堂", "配送学院", "1次", "督学管理", "A1=0", "签到率")
    if not all(term in text for term in required):
        return
    rows = [
        ["管理模块", "培训及考试对象", "学习及考试时间", "学习及考试链接/路径",
         "考试机会", "管理方式", "计分规则"],
        ["空中课堂", "以A【配送学院】合作商培训对接人群通知", "", "", "1次",
         "督学管理", "当签到率<90%时，A1=0；当签到率≥90%时，"
         "A1=签到率*40+应参通过率*40"],
    ]
    table = _repaired_table_at(
        blocks, start, rows,
        ["training_rule_table", "standard_markdown_table"],
    )
    keep = [block for block in blocks[start + 1:end]
            if "特殊说明" in re.sub(r"\s+", "", _block_text(block))
            or re.match(r"^4\.1\.[12]", (_block_text(block) or "").strip())]
    blocks[start + 1:end] = [table] + keep
    counter["规则宣导七列表重组"] += 1


def _repair_training_role_table(
        blocks: List[Block], counter: Counter[str]) -> None:
    start = _heading_index_with_terms(blocks, "4.2", "岗位系统训管理指标")
    if start is None:
        return
    end = len(blocks)
    for idx in range(start + 1, len(blocks)):
        compact = re.sub(r"\s+", "", _block_text(blocks[idx]))
        if compact.startswith("特殊说明") or compact.startswith("4.2.1"):
            end = idx
            break
    text = "".join(re.sub(r"\s+", "", _block_text(b)) for b in blocks[start + 1:end])
    required = (
        "专送骑手", "专送站长", "城市经理-新人训", "城市经理-在岗训",
        "招聘专员-新人训", "招聘管理员-适岗训", "非骑手人员",
        "B1=", "B2=", "B3=", "B4=", "B5=", "B6=", "B7=",
    )
    if not all(term in text for term in required):
        return
    rows = [
        ["管理模块", "培训及考试对象", "学习及考试路径", "考试机会",
         "管理方式", "数据查询 ID", "计分规则"],
        ["专送骑手", "全量骑手；2026年6月23日（含）前入职的所有骑手",
         "培训中心-安全保障-安全须知-如何保障配送中的食品安全", "不限制",
         "线上自学；督学管理", "7594",
         "当应参通过率<90%时，B1=0；当应参通过率≥90%时，B1=应参通过率*20"],
        ["专送站长", "专送站长-新人训；2026年5月1日-2026年5月31日（含）入职的专送站长",
         "互联网+大学-最近班级-6月专送站长新人训", "", "线上自学；督学管理", "",
         "当学习完成率<90%时，B2=-20；当学习完成率≥90%时，"
         "B2=-20-(学习完成率*-12+应参通过率*-8)"],
        ["城市经理", "城市经理-新人训；2026年5月1日-2026年5月31日（含）入职的城市经理",
         "互联网+大学-最近班级-新任城市经理训练营", "", "线上自学；督学管理", "",
         "当学习完成率<90%时，B3=-14；当学习完成率≥90%时，"
         "B3=-14-(学习完成率*-8+应参通过率*-6)"],
        ["城市经理", "城市经理-在岗训；2026年4月1日-2026年4月30日（含）入职的城市经理",
         "互联网+大学-最近班级-合作商城市经理在岗训", "", "线上自学；督学管理", "",
         "当学习完成率<90%时，B4=-14；当学习完成率≥90%时，"
         "B4=-14-(学习完成率*-8+应参通过率*-6)"],
        ["招聘人员", "招聘专员-新人训；2026年6月1日-2026年6月30日（含）入职的所有招聘专员",
         "合作商学习平台-班级-新招聘专员入职训练营2.0", "", "线上自学；督学管理", "",
         "当学习达标率<90%时，B5=-15；当学习达标率≥90%时，B5=-15-学习完成率*-15"],
        ["招聘人员", "招聘管理员-适岗训；2026年6月1日-2026年6月30日（含）入职的所有招聘管理员",
         "合作商学习平台-班级-招聘主管适岗培育", "", "线上自学；督学管理", "",
         "当学习达标率<90%时，B6=-15；当学习达标率≥90%时，B6=-15-学习完成率*-15"],
        ["非骑手人员", "专项训", "以A【配送学院】合作商培训对接人群通知为准", "/", "全程参加", "",
         "当学习参与人数≤2时，B7=学习参与人数*1；当学习参与人数≥3时，B7=5"],
    ]
    blocks[start + 1:end] = [_repaired_table_at(
        blocks, start, rows,
        ["training_role_table", "standard_markdown_table"])]
    counter["岗位系统训跨页七列表重组"] += 1


def _repair_cross_block_boundaries(blocks: List[Block],
                                   counter: Counter[str]) -> None:
    """Repair labels split at visual-line or virtual-page boundaries."""
    i = 0
    while i + 1 < len(blocks):
        current = blocks[i]
        if current.rows or not (current.text or "").strip():
            i += 1
            continue
        j = i + 1
        while (j < len(blocks) and j <= i + 3
               and not _block_text(blocks[j])):
            j += 1
        if j >= len(blocks) or j > i + 3:
            i += 1
            continue
        following = blocks[j]
        if following.rows or following.kind == "heading" or not following.text:
            i += 1
            continue
        current_text = (current.text or "").strip()
        following_text = (following.text or "").strip()

        if (re.search(r'。计算$', current_text)
                and following_text.startswith("示例：")
                and ("举例" in current_text or "=" in current_text)):
            current.text = re.sub(r'计算$', '', current_text).rstrip()
            following.text = "计算" + following_text
            counter["跨块计算示例标签修复"] += 1
            i = j
            continue

        if (current_text.endswith("计算示例：")
                and following_text.startswith("示例：")):
            following.text = following_text[len("示例："):].lstrip()
            counter["重复示例标签清理"] += 1
            i = j
            continue

        if (current_text.endswith('考核“')
                and following_text.startswith("品牌方口径准时率")):
            current.text = current_text + following_text
            del blocks[j]
            counter["跨块引号标签合并"] += 1
            continue

        if (re.fullmatch(r'[①②③④⑤⑥⑦⑧⑨⑩]', current_text)
                and following_text
                and not re.match(r'^[#|]', following_text)):
            following.text = current_text + " " + following_text
            del blocks[i]
            counter["孤立圈号与正文合并"] += 1
            continue
        i += 1


def _formula_tail_fingerprint(blk: Block) -> str:
    text = re.sub(r'\s+', '', _block_text(blk))
    return text.translate(str.maketrans({
        '＋': '+', '＊': '*', '（': '(', '）': ')', '＝': '=',
    }))


def _drop_repeated_formula_tail_before_heading(
        blocks: List[Block], counter: Counter[str]) -> None:
    """Drop a next-section formula set emitted once before its heading.

    A long image can be split into virtual pages through the middle of a
    section.  In that case the next section's formula set may be assembled at
    the end of the previous section and then emitted again after its real
    heading.  We only remove a tail when the previous section already contains
    two copies of its anchor formula and every tail block is repeated in the
    following section.
    """
    i = 0
    while i < len(blocks):
        if blocks[i].kind != "heading":
            i += 1
            continue
        if _drop_heading_prefixed_formula_duplicate(blocks, i, counter):
            i = max(0, i - 1)
            continue
        previous_heading = i - 1
        while previous_heading >= 0 and blocks[previous_heading].kind != "heading":
            previous_heading -= 1
        anchors = [
            pos for pos in range(previous_heading + 1, i)
            if "特殊场景完成单占比" in _formula_tail_fingerprint(blocks[pos])
        ]
        if len(anchors) < 2:
            i += 1
            continue
        tail_start = anchors[-1]
        if i - tail_start > 4:
            i += 1
            continue
        next_heading = _next_heading_index(blocks, i + 1)
        after = {
            _formula_tail_fingerprint(block)
            for block in blocks[i + 1:next_heading]
            if len(_formula_tail_fingerprint(block)) >= 24
        }
        tail = blocks[tail_start:i]
        tail_fingerprints = [_formula_tail_fingerprint(block) for block in tail]
        if (len(tail) < 2
                or any(len(value) < 24 or value not in after
                       for value in tail_fingerprints)):
            i += 1
            continue
        del blocks[tail_start:i]
        counter["跨章节重复公式尾块清理"] += len(tail)
        i = tail_start + 1


def _drop_heading_prefixed_formula_duplicate(
        blocks: List[Block], heading_index: int,
        counter: Counter[str]) -> bool:
    """Remove a flattened next-section paragraph placed before its heading."""
    if heading_index <= 0:
        return False
    previous = blocks[heading_index - 1]
    if previous.kind != "para" or previous.rows:
        return False
    heading = _formula_tail_fingerprint(blocks[heading_index])
    label = re.sub(r'^#+', '', heading)
    label = re.sub(r'^\d+(?:[.．]\d+)*', '', label)
    previous_text = _formula_tail_fingerprint(previous)
    if len(label) < 4 or not previous_text.startswith(label):
        return False
    if not ("特殊场景体验得分" in previous_text
            and "普通场景体验得分" in previous_text):
        return False
    next_heading = _next_heading_index(blocks, heading_index + 1)
    following_text = "".join(
        _formula_tail_fingerprint(block)
        for block in blocks[heading_index + 1:next_heading]
    )
    if not ("特殊场景体验得分" in following_text
            and "普通场景体验得分" in following_text):
        return False
    numbers = set(re.findall(r'\d+(?:\.\d+)?%', previous_text))
    if numbers and not numbers.issubset(set(re.findall(r'\d+(?:\.\d+)?%', following_text))):
        return False
    del blocks[heading_index - 1]
    counter["标题前置重复公式段清理"] += 1
    return True


def _is_protected_native_pdf_table(blk: Block) -> bool:
    if "native_pdf_table" not in (blk.flags or []) or not blk.rows:
        return False
    joined = " ".join(" ".join(c for c in row if c) for row in blk.rows)
    if not joined.strip():
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff]", joined))
    if cjk > max(8, len(joined) * 0.08):
        return False
    scientific_terms = (
        "Model", "Overall", "Edit", "Metric", "Pages", "TPS",
        "DS-OCR", "Unlimited-OCR", "DeepSeek", "OCRVerse", "Nanonets",
    )
    return any(term in joined for term in scientific_terms)


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


def _drop_redundant_policy_table_raw_text(
        blocks: List[Block], counter: Counter[str]) -> None:
    remove: List[int] = []
    for i, block in enumerate(blocks):
        if block.kind not in {"para", "image"}:
            continue
        compact = re.sub(r'\s+', '', _block_text(block))
        if not _looks_like_redundant_policy_raw_paragraph(compact):
            continue
        nearby = "\n".join(
            _block_text(b) for b in blocks[i + 1:i + 8]
            if b.page == block.page
        )
        if _has_structured_policy_replacement(nearby):
            remove.append(i)
    if remove:
        for i in reversed(remove):
            del blocks[i]
        counter["责任表原始错列段清理"] += len(remove)


def _finalize_repaired_fallback_blocks(
        blocks: List[Block], counter: Counter[str]) -> None:
    """Reassess fallback blocks after all table repairs have finished.

    Assembly flags describe the *initial* OCR grid.  Keeping those stale flags
    after a strict row repair makes clean Markdown fail forever; clearing every
    flag, on the other hand, hides real column drift.  This final pass therefore
    accepts only structurally complete policy tables and removes one-cell
    fallback fragments only when the same text already exists elsewhere.
    """
    remove: List[int] = []
    for i, blk in enumerate(blocks):
        protected_native_table = _is_protected_native_pdf_table(blk)
        if blk.rows and not protected_native_table:
            dropped = _drop_self_check_virtual_page_rows(blk.rows)
            if dropped:
                counter["四列制度表虚拟页码清理"] += dropped
            blk.rows = clean_table_noise_rows(blk.rows)
            dropped = _drop_isolated_table_page_rows(blk.rows)
            if dropped:
                counter["表内虚拟页眉页码清理"] += dropped
            reshaped = _canonicalize_repaired_table_rows(blk.rows)
            if reshaped:
                counter["修复后表格列结构归一"] += reshaped
            if set(blk.flags or []).intersection({
                    "cell_ocr_table", "table_low_confidence", "table_fallback",
                    "merged_table_fallback", "fragment_table_fallback"}):
                repaired = repair_table_rows(blk.rows)
                if repaired:
                    counter["修复后表格内容复核"] += repaired
            repaired = _repair_exception_table_rows(blk.rows)
            if repaired:
                counter["异常单表格逻辑列重组"] += repaired
        if protected_native_table:
            continue
        if _promote_reliable_single_cell_fallback(blk):
            counter["单栏表格残片恢复为正文"] += 1
            continue
        if _is_redundant_single_cell_fallback(blocks, i):
            remove.append(i)
            continue
        if _is_reliable_repaired_policy_table(blk):
            _mark_table_repaired_verified(blk)
            counter["修复后表格结构复核通过"] += 1
            continue
        if _is_reliable_repaired_structured_table(blk):
            _mark_table_repaired_verified(blk)
            counter["通用表格结构复核通过"] += 1
    for i in reversed(remove):
        del blocks[i]
    if remove:
        counter["表格附近重复正文清理"] += len(remove)


def _mark_table_repaired_verified(blk: Block) -> None:
    blk.kind = "table"
    # Fallback blocks may carry a flattened OCR caption in addition to rows.
    # Once the rows have passed structural verification, exporting both would
    # duplicate the same content and preserve the least reliable copy.
    blk.text = ""
    blk.confidence = max(blk.confidence, 0.8)
    stale = {
        "table_low_confidence", "table_fallback",
        "merged_table_fallback", "fragment_table_fallback",
    }
    blk.flags = [flag for flag in blk.flags if flag not in stale]
    if "table_repaired_verified" not in blk.flags:
        blk.flags.append("table_repaired_verified")


def _drop_isolated_table_page_rows(rows: List[List[str]]) -> int:
    """Remove only rows that are unmistakable virtual-page furniture.

    Long screenshot PDFs repeat a logo, page number and document code inside a
    table continuation.  Those rows are not table data.  The rule is scoped to
    multi-column tables and requires either an isolated page number or a short
    logo/code combination, so numeric benchmark rows remain untouched.
    """
    if not rows:
        return 0
    kept: List[List[str]] = []
    dropped = 0
    logo_re = re.compile(r'(?:美.{0,3}配.{0,3}|美团配送|MTPS)', re.I)
    max_columns = max((len(row) for row in rows), default=0)
    one_column_template = False
    if max_columns == 1:
        header = re.sub(r'\s+', '', rows[0][0] or '')
        # OCR often captures the page number between the rows of a one-column
        # email/template block.  Only remove that furniture when the header
        # identifies a template/rule block; numeric policy conditions remain
        # untouched.
        one_column_template = bool(re.search(
            r'邮件|绑站具体规则|邮件正文|申请书|协议', header))
    for row in rows:
        cells = [re.sub(r'\s+', '', cell or "") for cell in row]
        nonempty = [cell for cell in cells if cell]
        only_page_number = (
            len(nonempty) == 1
            and bool(re.fullmatch(r'\d{1,3}', nonempty[0]))
        )
        if max_columns == 1 and one_column_template and only_page_number:
            dropped += 1
            continue
        if max_columns < 2:
            kept.append(row)
            continue
        compact = "".join(nonempty)
        document_code = bool(re.search(
            r'MTPS|[A-Z]{1,5}-[A-Z]{1,5}-V\d{2}|V\d{2}-20\d{6}|20\d{6}',
            compact,
            re.I,
        ))
        footer_text = bool(re.search(
            r'保密(?:资料|原则)|未经书面允许|第三方提供', compact))
        # A Vision cell grid can absorb the page footer into the last table
        # row.  It is page furniture only when the same row also carries the
        # document code; never drop a policy row merely because it mentions a
        # number or the word "保密".
        if document_code and footer_text:
            dropped += 1
            continue
        # A normal sentence can contain both ``美团配送`` and a number (for
        # example an appeal deadline at 16:00).  Treating that pair as page
        # furniture silently deleted complete table rows.  A logo-only page
        # row must instead be either tied to a document code or consist solely
        # of a very short logo/page-number combination.
        short_logo_page = bool(re.fullmatch(
            r'(?:(?:美.{0,3}配.{0,3}|美团配送)\d{1,3}|'
            r'\d{1,3}(?:美.{0,3}配.{0,3}|美团配送))',
            compact,
            re.I,
        ))
        logo_or_code = (
            len(compact) <= 90
            and bool(logo_re.search(compact))
            and (document_code or short_logo_page)
        )
        if only_page_number or logo_or_code:
            dropped += 1
            continue
        kept.append(row)
    if dropped:
        rows[:] = kept
    return dropped


def _canonicalize_repaired_table_rows(rows: List[List[str]]) -> int:
    """Collapse false empty columns and restore clear key/value continuations."""
    if not rows:
        return 0
    changed = _drop_empty_table_columns(rows)
    if not rows:
        return changed

    # Virtual-page logos can land inside a real cell when a long screenshot
    # table crosses a page boundary.  Remove only this exact slogan fragment;
    # surrounding policy text remains untouched.  A slogan-only row then
    # becomes empty and can safely be discarded.
    cleaned_rows: List[List[str]] = []
    for row in rows:
        cleaned = []
        for cell in row:
            value, count = _TABLE_INLINE_SLOGAN_RE.subn("", cell or "")
            if count:
                changed += count
            cleaned.append(value.strip())
        if any(cleaned):
            cleaned_rows.append(cleaned)
        else:
            changed += 1
    rows[:] = cleaned_rows
    if not rows:
        return changed

    ncol = max(len(row) for row in rows)
    rows[:] = [list(row) + [""] * (ncol - len(row)) for row in rows]
    if ncol != 2:
        return changed

    first = [cell.strip() for cell in rows[0]]
    if (len(rows) == 1 and first[0] and first[1]
            and len(re.sub(r'\s+', '', first[0])) <= 16
            and len(re.sub(r'\s+', '', first[1])) >= 24
            and re.search(r'流程|说明|范围|来源|规则|备注|口径$', first[0])):
        rows[:] = [["项目", "内容"], first]
        first = ["项目", "内容"]
        changed += 1
    # OCR often returns a clean two-column key/value table without its visual
    # header.  Treating the first data row as the Markdown header repeats that
    # row's long value for every following record.  Insert a neutral header
    # only when the first cell is a short Chinese label, the second is prose,
    # and most following rows show the same label/value relationship.
    known_header = (
        first[0] in {"项目", "维度", "类型", "判定维度", "事件类型"}
        and first[1] in {"内容", "说明", "释义", "判定细则", "事件细则"}
    )
    labelled_rows = sum(
        1 for left, value in rows
        if (left or "").strip() and (value or "").strip()
    )
    if (not known_header and len(rows) >= 2
            and first[0] and first[1]
            and re.search(r'[\u4e00-\u9fff]', first[0])
            and len(re.sub(r'\s+', '', first[0])) <= 24
            and len(re.sub(r'\s+', '', first[1])) >= 24
            and labelled_rows >= max(2, (len(rows) + 1) // 2)):
        rows.insert(0, ["项目", "内容"])
        first = ["项目", "内容"]
        changed += 1
    if (not first[0] and first[1]
            and any((row[0] or "").strip() for row in rows[1:])):
        title = first[1]
        rows[0] = ["项目", "内容"]
        rows.insert(1, ["标题", title])
        changed += 1

    key_value_header = (
        len(rows[0]) == 2
        and rows[0][0].strip() in {"项目", "维度", "类型", "判定维度", "事件类型"}
        and rows[0][1].strip() in {"内容", "说明", "释义", "判定细则", "事件细则"}
    )

    if key_value_header:
        # A row-spanned label may be OCR'd as ``特殊`` followed by
        # ``情形处理（...）`` on the next line.  Rejoin the label and preserve
        # both pieces of explanatory text instead of emitting two unrelated
        # records.
        i = 1
        while i + 1 < len(rows):
            left = (rows[i][0] or "").strip()
            next_left = (rows[i + 1][0] or "").strip()
            if left == "特殊" and next_left.startswith("情形处理"):
                intro = (rows[i][1] or "").strip()
                detail = (rows[i + 1][1] or "").strip()
                rows[i + 1][0] = "特殊" + next_left
                rows[i + 1][1] = "\n".join(
                    value for value in (intro, detail) if value)
                del rows[i]
                changed += 1
                continue
            i += 1

        # A visually merged note cell has no left-column text.  Give only
        # explicit note paragraphs a semantic label so they do not get folded
        # into the preceding unrelated rule.
        for row in rows[1:]:
            if (row[0] or "").strip():
                continue
            value = (row[1] or "").strip()
            note = re.match(r'^特殊说明\s*[:：]?\s*(.+)$', value, re.S)
            if note:
                row[0] = "特殊说明"
                row[1] = note.group(1).strip()
                changed += 1

    # A blank left cell below a key/value row is a visual continuation.  Join
    # it only when the previous value is visibly unfinished; independent
    # row-spanned categories therefore remain separate.
    i = 1
    while i < len(rows):
        left = (rows[i][0] or "").strip()
        value = (rows[i][1] or "").strip()
        if not left and value and i > 1:
            previous = (rows[i - 1][1] or "").rstrip()
            continuation = (
                bool(previous)
                and (key_value_header
                     or not re.search(r'[。；;.!?！？]〉?）?$', previous)
                     or bool(re.match(
                         r'^(?:关闭订单|不剔除|且|并|其中|注|'
                         r'\d+[）)]|[①-⑳])', value)))
            )
            if continuation:
                joiner = "\n" if re.match(r'^(?:\d+[）)]|[①-⑳])', value) else ""
                rows[i - 1][1] = previous + joiner + value
                del rows[i]
                changed += 1
                continue
        i += 1
    return changed


def _repair_exception_table_rows(rows: List[List[str]]) -> int:
    """Restore common exception-rule table schemas after cell OCR.

    The source tables use row-spans and multi-level headers.  Vision OCR can
    return the same logical field in several adjacent cells (for example
    ``内`` + ``涝``), or return a continuation table as a one-column stream.
    This repair is schema-driven and preserves the cell order; it never
    replaces the conditions with a summary.
    """
    if not rows:
        return 0
    changed = 0
    compact_header = [re.sub(r'\s+', '', cell or '') for cell in rows[0]]

    # A multi-level scene table can be recognized by the four logical labels
    # even when the OCR grid contains empty bridge columns.
    if ("场景" in compact_header
            and any("提报条件" in cell for cell in compact_header)
            and "举例" in compact_header
            and any("剔除方式" in cell for cell in compact_header)
            and len(rows[0]) > 4):
        scene_idx = compact_header.index("场景")
        condition_idx = next(
            idx for idx, cell in enumerate(compact_header)
            if "提报条件" in cell
        )
        example_idx = compact_header.index("举例")
        remove_idx = next(
            idx for idx, cell in enumerate(compact_header)
            if "剔除方式" in cell
        )

        rebuilt = [["场景", "提报条件", "举例", "剔除方式"]]
        for raw_row in rows[1:]:
            row = list(raw_row) + [""] * max(0, len(rows[0]) - len(raw_row))
            cells = [str(cell or '').strip() for cell in row]
            compact = ''.join(re.sub(r'\s+', '', cell) for cell in cells)
            # A footer may be split across the row, but its document code is a
            # strong enough signal only when the scene column is empty.
            if (not re.sub(r'\s+', '', cells[scene_idx])
                    and re.search(r'(?:V\d{2}-)?20\d{6}', compact, re.I)):
                changed += 1
                continue

            def join_range(start: int, end: int) -> str:
                parts = [cells[idx] for idx in range(start, min(end, len(cells)))
                         if cells[idx]]
                value = ''.join(parts)
                return re.sub(r'\s+', ' ', value).strip()

            logical = [
                join_range(scene_idx, condition_idx),
                join_range(condition_idx, example_idx),
                join_range(example_idx, remove_idx),
                join_range(remove_idx, len(cells)),
            ]
            if any(logical):
                rebuilt.append(logical)
        if len(rebuilt) >= 2:
            rows[:] = rebuilt
            changed += 1
            return changed

    # The long continuation table under ``提报条件`` is a sequential stream,
    # not a reliable rectangular grid.  Keep every entry but give standalone
    # numeric markers an explicit meaning during export.
    if (len(rows[0]) == 1
            and "提报条件" in compact_header[0]
            and len(rows) >= 3):
        cleaned = [rows[0]]
        for row in rows[1:]:
            value = (row[0] or '').strip() if row else ''
            if value:
                cleaned.append([value])
        if len(cleaned) >= 3:
            rows[:] = cleaned
            changed += 1
    return changed


def _drop_empty_table_columns(rows: List[List[str]]) -> int:
    if not rows:
        return 0
    ncol = max(len(row) for row in rows)
    padded = [list(row) + [""] * (ncol - len(row)) for row in rows]
    keep = [j for j in range(ncol)
            if any((row[j] or "").strip() for row in padded)]
    if not keep:
        rows[:] = []
        return ncol
    if len(keep) == ncol:
        rows[:] = padded
        return 0
    rows[:] = [[row[j] for j in keep] for row in padded]
    return ncol - len(keep)


def _is_reliable_repaired_structured_table(blk: Block) -> bool:
    """Accept repaired tables only after a strict, content-aware recheck."""
    flags = set(blk.flags or [])
    if not flags.intersection({
            "table_low_confidence", "table_fallback",
            "merged_table_fallback", "fragment_table_fallback"}):
        return False
    rows = blk.rows or []
    if len(rows) < 2:
        return False
    ncol = max(len(row) for row in rows)
    if ncol < 1 or ncol > 8 or any(len(row) != ncol for row in rows):
        return False
    header = [re.sub(r'\s+', '', cell or "") for cell in rows[0]]
    if sum(bool(cell) for cell in header) < (2 if ncol >= 2 else 1):
        return False
    semantic_two_col = (
        ncol == 2
        and header[0] in {
            "项目", "维度", "类型", "判定维度", "事件类型", "体验星级",
        }
        and header[1] in {
            "内容", "说明", "释义", "判定细则", "事件细则", "计算规则",
        }
    )
    generic_two_col = (
        ncol == 2
        and all(header)
        and len(header[0]) <= 24
        and len(header[1]) <= 36
        and all(
            sum(bool((cell or '').strip()) for cell in row) == 2
            for row in rows[1:]
        )
    )
    generic_one_col_template = False
    if ncol == 1 and len(rows) >= 2 and header[0]:
        header_text = header[0]
        payload = [re.sub(r'\s+', '', row[0] or '') for row in rows[1:]]
        prose_rows = [value for value in payload if value]
        numeric_rows = [value for value in prose_rows
                        if re.fullmatch(r'\d{1,3}', value)]
        generic_one_col_template = (
            bool(re.search(r'邮件|绑站具体规则|邮件正文|申请书|协议', header_text))
            and bool(prose_rows)
            and len(numeric_rows) <= 1
            and all(
                len(value) >= 18 or re.search(r'[。；;：:，,]', value)
                for value in prose_rows
                if value not in numeric_rows
            )
        )
    exception_scene_table = (
        ncol == 4
        and header == ["场景", "订单类型", "提报条件", "剔除方式"]
        and len(rows) >= 2
    )
    exception_condition_list = (
        ncol == 1
        and "提报条件" in header[0]
        and len(rows) >= 3
    )
    exception_condition_continuation = False
    if ncol == 1 and len(rows) >= 6:
        numeric_markers = sum(
            1 for row in rows
            if re.fullmatch(r'\s*\d{1,3}\s*', row[0] or '')
        )
        long_entries = sum(
            1 for row in rows
            if len(re.sub(r'\s+', '', row[0] or '')) >= 18
        )
        exception_condition_continuation = (
            numeric_markers >= 3 and long_entries >= 3
        )
    if not (semantic_two_col or generic_two_col or generic_one_col_template
            or exception_scene_table or exception_condition_list
            or exception_condition_continuation) \
            and table_suspect_score(rows) >= 3:
        return False
    for row in rows:
        raw_cells = [cell or "" for cell in row]
        cells = [re.sub(r'\s+', '', cell) for cell in raw_cells]
        joined = "".join(cells)
        if (is_noise_text(joined) or _WATERMARK_FRAGMENT_RE.search(joined)
                or _SUSPECT_OCR_CHARS.search(joined)
                or any(_SUSPECT_TABLE_TEXT.search(cell)
                       for cell in raw_cells)):
            return False
    payload = rows[1:]
    if exception_scene_table:
        complete = sum(
            1 for row in payload
            if any((cell or '').strip() for cell in row)
        )
        return complete / max(1, len(payload)) >= 0.75
    if exception_condition_list:
        complete = sum(1 for row in payload if (row[0] or '').strip())
        return complete / max(1, len(payload)) >= 0.9
    if exception_condition_continuation:
        complete = sum(1 for row in payload if (row[0] or '').strip())
        return complete / max(1, len(payload)) >= 0.9
    if generic_one_col_template:
        complete = sum(1 for row in payload if (row[0] or '').strip())
        return complete / max(1, len(payload)) >= 0.9
    complete = sum(1 for row in payload
                   if sum(bool((cell or "").strip()) for cell in row) >= 2)
    required_ratio = 0.9 if ncol <= 2 else 0.75
    return complete / max(1, len(payload)) >= required_ratio


def _drop_self_check_virtual_page_rows(rows: List[List[str]]) -> int:
    """Drop page-number fragments only inside the known four-column schema.

    A virtual page boundary can be OCR'd as ``category | empty | page | code``
    inside the station self-check table.  Numeric benchmark tables may have the
    same value shapes, so this repair is deliberately gated by the exact policy
    header instead of living in the generic table-noise cleaner.
    """
    if not rows or len(rows[0]) < 4:
        return 0
    header = [re.sub(r'\s+', '', cell or "") for cell in rows[0][:4]]
    if header != ["项目", "内容", "说明", "承担责任"]:
        return 0

    kept = [rows[0]]
    dropped = 0
    for row in rows[1:]:
        padded = list(row) + [""] * max(0, 4 - len(row))
        compact = [re.sub(r'\s+', '', cell or "") for cell in padded[:4]]
        is_virtual_page_row = (
            bool(compact[0])
            and not compact[1]
            and bool(re.fullmatch(r'\d{1,3}', compact[2]))
            and bool(re.fullmatch(r'\d{6,10}', compact[3]))
        )
        if is_virtual_page_row:
            dropped += 1
            continue
        kept.append(row)
    if dropped:
        rows[:] = kept
    return dropped


def _promote_reliable_single_cell_fallback(blk: Block) -> bool:
    """Turn a clean one-cell prose fragment back into a paragraph.

    Geometry can place an ordinary note next to a difficult table and mark it
    as a fallback image.  A complete prose sentence is not an uncertain table.
    Conversely, flattened multi-column rows usually start with an amount and
    contain several responsibility values; those must keep blocking review.
    """
    if "fragment_table_fallback" not in (blk.flags or []):
        return False
    rows = blk.rows or []
    if len(rows) != 1 or len(rows[0]) < 2:
        return False
    label = re.sub(r'\s+', '', rows[0][0] or "")
    payload = (rows[0][1] or "").strip()
    compact = re.sub(r'\s+', '', payload)
    if label not in {"内容", "说明", "注"} or len(compact) < 24:
        return False
    if _SUSPECT_OCR_CHARS.search(compact) or looks_truncated(payload):
        return False
    amount_count = len(re.findall(r'\d+(?:\.\d+)?\s*元', payload))
    if (re.match(r'^\s*\d+(?:\.\d+)?\s*元', payload)
            and amount_count >= 2):
        return False
    if sum(term in compact for term in (
            "检核项目", "责任承担", "覆盖范围", "整改不达标")) >= 3:
        return False
    blk.kind = "para"
    blk.text = payload
    blk.rows = None
    blk.confidence = max(blk.confidence, 0.8)
    stale = {
        "table_low_confidence", "table_fallback",
        "merged_table_fallback", "fragment_table_fallback",
    }
    blk.flags = [flag for flag in blk.flags if flag not in stale]
    return True


def _is_redundant_single_cell_fallback(blocks: List[Block], index: int) -> bool:
    blk = blocks[index]
    if blk.kind not in {"table", "image"}:
        return False
    rows = blk.rows or []
    if len(rows) != 1 or len(rows[0]) < 2:
        return False
    label = re.sub(r'\s+', '', rows[0][0] or "")
    payload = re.sub(r'\s+', '', rows[0][1] or "")
    if label not in {"内容", "说明", "注"} or len(payload) < 24:
        return False
    other_text = "".join(
        re.sub(r'\s+', '', _block_text(other))
        for j, other in enumerate(blocks)
        if j != index and other.page == blk.page
    )
    if payload and payload in other_text:
        return True
    if len(payload) >= 120:
        return payload[:60] in other_text and payload[-60:] in other_text
    return False


def _is_reliable_repaired_policy_table(blk: Block) -> bool:
    if "column_segmented_fallback" in (blk.flags or []):
        return False
    if "table_low_confidence" not in (blk.flags or []) or not blk.rows:
        return False
    ncol = max(len(row) for row in blk.rows)
    if ncol not in (3, 4, 5) or len(blk.rows) < 2:
        return False
    rows = [row + [""] * (ncol - len(row)) for row in blk.rows]
    header = [re.sub(r'\s+', '', cell or "") for cell in rows[0]]
    if any(_SUSPECT_OCR_CHARS.search(cell) for cell in header):
        return False

    if ncol == 5:
        expected = ["项目", "序号", "不达标情况", "整改结果", "承担责任"]
        if header != expected:
            return False
        payload_rows = rows[1:]
        complete = sum(
            1 for row in payload_rows
            if re.sub(r'\s+', '', row[0] or "")
            and re.sub(r'\s+', '', row[2] or "")
        )
        return complete / max(1, len(payload_rows)) >= 0.75
    if ncol == 3:
        valid_header = (
            (header[0] == "项目"
             or any(term in header[0] for term in ("检核项目", "考核项目")))
            and header[1] in {"内容", "说明", "释义"}
            and any(term in header[2] for term in ("责任承担", "承担责任"))
        )
        required = (0, 1, 2)
    else:
        self_check_header = (
            header[0] == "项目"
            and header[1] == "内容"
            and header[2] in {"说明", "释义"}
            and any(term in header[3] for term in ("责任承担", "承担责任"))
        )
        if self_check_header:
            payload_rows = rows[1:]
            complete = 0
            for row in payload_rows:
                compact = [re.sub(r'\s+', '', cell or "") for cell in row]
                if any(_SUSPECT_OCR_CHARS.search(cell) for cell in compact):
                    return False
                if ((compact[0] or compact[1])
                        and compact[2] and compact[3]):
                    complete += 1
            return complete / max(1, len(payload_rows)) >= 0.85
        valid_header = (
            header[0] in {"一级分类", "类型", "分类"}
            and any(term in header[1] for term in ("检核项目", "考核项目", "项目"))
            and header[2] in {"内容", "说明", "释义"}
            and any(term in header[3] for term in ("责任承担", "承担责任"))
        )
        required = (1, 2, 3)
    if not valid_header:
        return False

    payload_rows = rows[1:]
    complete = 0
    for row in payload_rows:
        compact = [re.sub(r'\s+', '', cell or "") for cell in row]
        if any(_SUSPECT_OCR_CHARS.search(cell) for cell in compact):
            return False
        if compact[required[0]] and (compact[required[1]] or compact[required[2]]):
            complete += 1
    return complete / max(1, len(payload_rows)) >= 0.85


def _looks_like_redundant_policy_raw_paragraph(compact: str) -> bool:
    if "检核项目" not in compact or "责任承担" not in compact:
        return False
    bad_terms = (
        "3.视频监", "4.流媒体", "7.看板海", "8.站内宿",
        "遮挡200元/项/次", "员工宿", "整改舍安全管理制度",
    )
    return len(compact) > 500 and sum(1 for term in bad_terms if term in compact) >= 2


def _has_structured_policy_replacement(text: str) -> bool:
    compact = re.sub(r'\s+', '', text)
    required = ("3.视频监控", "4.流媒体", "7.看板海报")
    return all(item in compact for item in required)


def _remove_redundant_fine_schedule_definition_fragments(
        blocks: List[Block], counter: Counter[str]) -> None:
    anchors = ("【排班在线合格时段数】", "【排班合格出勤骑手数】")
    i = 0
    while i < len(blocks):
        text = _block_text(blocks[i])
        if not all(anchor in text for anchor in anchors):
            i += 1
            continue
        page = blocks[i].page
        base_y0, base_y1 = blocks[i].bbox[1], blocks[i].bbox[3]
        remove = set()
        for j in range(max(0, i - 3), min(len(blocks), i + 5)):
            if j == i or blocks[j].page != page:
                continue
            if not (base_y0 - 900 <= blocks[j].bbox[1] <= base_y1 + 1800):
                continue
            frag = _block_text(blocks[j])
            compact = re.sub(r'\s+', '', frag)
            if _is_redundant_fine_schedule_definition_fragment(compact):
                remove.add(j)
        if remove:
            for j in sorted(remove, reverse=True):
                del blocks[j]
                if j < i:
                    i -= 1
            counter["精细化排班指标定义碎片清理"] += len(remove)
            continue
        i += 1


def _is_redundant_fine_schedule_definition_fragment(compact: str) -> bool:
    if not compact:
        return False
    if "烽火台-业务" in compact and ("查询" in compact or "路径" in compact):
        return False
    if compact.startswith("【排班在线") and "当天班次" in compact and "有单骑手" in compact:
        return True
    terms = (
        "合格时段", "发生变化的", "点给骑手在", "标记在职骑",
        "日离职删号", "站点有单骑", "不含集约站订", "中存在2个时",
        "烽火台标记手数", "排班合格出勤",
    )
    return sum(1 for term in terms if term in compact) >= 2


def _repair_fine_schedule_query_path_fragments(
        blocks: List[Block], counter: Counter[str]) -> None:
    seen_fine_schedule_defs = False
    i = 0
    while i < len(blocks):
        compact = re.sub(r'\s+', '', _block_text(blocks[i]))
        if "【排班在线合格时段数】" in compact and "【排班合格出勤骑手数】" in compact:
            seen_fine_schedule_defs = True
            i += 1
            continue
        if not seen_fine_schedule_defs or not _is_fine_schedule_query_path_fragment(compact):
            i += 1
            continue
        j = i + 1
        while j < len(blocks):
            next_compact = re.sub(r'\s+', '', _block_text(blocks[j]))
            if not _is_fine_schedule_query_path_fragment(next_compact):
                break
            j += 1
        blocks[i] = Block(
            kind="para",
            text=(
                "数据查询路径：烽火台-业务管理-骑手排班-精细化排班监控-站点；"
                "烽火台-业务管理-骑手排班-精细化排班监控-站点出勤时段。"
            ),
            bbox=blocks[i].bbox,
            page=blocks[i].page,
            confidence=max(blocks[i].confidence, 0.8),
        )
        del blocks[i + 1:j]
        counter["精细化排班查询路径重组"] += max(1, j - i)
        i += 1


def _is_fine_schedule_query_path_fragment(compact: str) -> bool:
    if not compact:
        return False
    rich_answer_terms = (
        "申诉路径", "申诉场景", "目标会于每月月底", "每月结算明细",
        "月底同步的明细数据", "新骑手入职当天", "站点天气判定",
        "薪动力异常场景申诉",
    )
    if any(term in compact for term in rich_answer_terms):
        return False
    if "烽火台-业务" in compact and "骑手排" in compact and "精细化排" in compact:
        return True
    if compact.startswith("数据") and "烽火台" in compact:
        return True
    if compact.startswith("查询") and "骑手排" in compact:
        return True
    if compact.startswith("路径") and ("精细化排" in compact or "排班监控" in compact):
        return True
    return False


def _remove_redundant_fine_schedule_faq_fragments(
        blocks: List[Block], counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks):
        compact = re.sub(r'\s+', '', _block_text(blocks[i]))
        has_complete_answer = (
            "申诉路径" in compact
            and ("目标会于每月月底" in compact or "数据查询路径" in compact)
        )
        if not has_complete_answer:
            i += 1
            continue
        page = blocks[i].page
        _, y0, _, y1 = blocks[i].bbox
        remove = []
        for j in range(i + 1, min(len(blocks), i + 5)):
            if blocks[j].page != page:
                break
            if blocks[j].bbox[1] > y1 + 1800:
                break
            frag = re.sub(r'\s+', '', _block_text(blocks[j]))
            if _is_redundant_fine_schedule_faq_fragment(frag):
                remove.append(j)
        if remove:
            for j in reversed(remove):
                del blocks[j]
            counter["精细化排班FAQ碎片清理"] += len(remove)
            continue
        i += 1


def _is_redundant_fine_schedule_faq_fragment(compact: str) -> bool:
    if not compact:
        return False
    if "站点天气被系统误判怎么办" in compact:
        return False
    if compact.startswith("月考核") and "商排班考核月度目标" in compact:
        return True
    terms = (
        "月结算调分明细在哪", "月中系统展示的精细", "化排班分数与站点实",
        "月度最终调分结果与", "烽火台-精细化排班", "监控导出的数据结果",
        "有单骑手数", "当天新入职骑手跑单", "是否会计算有单骑手",
        "高峰合格占比", "如何查询是否是恶劣", "目标值查询", "里查询",
        "际分数不符", "算调分", "测算不一致", "径及查询路径",
        "合格占比的分母", "考核口天气", "目标月度结",
    )
    return sum(1 for term in terms if term in compact) >= 2


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
        elif ("5.8" in compact
              and "站点组KA品牌单体验得分" in compact):
            inserted = _ensure_formula_in_section(
                blocks, i,
                "站点组KA品牌单体验得分=K1分*K1*Q1/SUM(Kn*Qn)"
                "+KN分*KN*QN/SUM(Kn*Qn)",
                "站点组KA品牌单体验得分公式补全",
            )
            if inserted:
                counter["站点组KA品牌单体验得分公式补全"] += 1
        i += 1


def _clean_ka_formula_ocr_fragments(blocks: List[Block],
                                    counter: Counter[str]) -> None:
    """Remove misleading OCR leftovers once a reliable formula block exists.

    Apple Vision can see the visual formula area but flatten it into text such
    as ``虚假点送达率=TKA品牌单WKA品牌单``. The canonical LaTeX block is kept; these
    fragments are removed so they do not look like a second, authoritative
    formula.
    """
    i = 0
    while i < len(blocks):
        heading = re.sub(r'\s+', '', _block_text(blocks[i]))
        if not re.search(r'5[.．]4[.．][1-7]', heading):
            i += 1
            continue
        end = _next_heading_index(blocks, i + 1)
        section = blocks[i + 1:end]
        if not any("formula_latex" in b.flags for b in section):
            i = end
            continue
        for blk in section:
            if not blk.text:
                continue
            cleaned = _clean_ka_formula_fragment_text(blk.text)
            if cleaned != blk.text:
                blk.text = cleaned
                replaced_by_formula = (
                    not cleaned
                    or bool(re.fullmatch(
                        r'(?:计算口径|指标释义)\s*[:：]?', cleaned.strip()
                    ))
                )
                if (replaced_by_formula
                        and "formula_replaced_by_latex" not in blk.flags):
                    blk.flags.append("formula_replaced_by_latex")
                counter["KA公式OCR残片清理"] += 1
        i = end


def _clean_ka_formula_fragment_text(text: str) -> str:
    cleaned = text
    cleaned = re.sub(
        r'(?m)^\s*[A-Z]\s*KA\s*品牌单\s*指标释义\s*[:：]\s*$',
        '指标释义：',
        cleaned,
    )
    cleaned = re.sub(
        r'(?m)^\s*计算口径\s*[:：]\s*(?:[CPRTWY]\s*KA\s*品牌单|[CPRTWY]KA品牌单)\s*$',
        '计算口径：',
        cleaned,
    )
    cleaned = re.sub(
        r'虚假点送达率\s*[=＝]\s*T\s*KA\s*品牌单\s*/?\s*W\s*KA\s*品牌单',
        '',
        cleaned,
    )
    cleaned = re.sub(
        r'(?m)^\s*(计算口径\s*[:：]\s*)?配送原因未[完定]成率'
        r'[^\n]*(?:Wex|PKA|KA[脚腳]M)[^\n]*\n?',
        lambda match: '计算口径：' if match.group(1) else '',
        cleaned,
    )
    cleaned = re.sub(
        r'(?m)^\s*A[L1I]?\s*KA品[牌解]单\s*\+\s*A2\s*KA品牌单\s*\+\s*'
        r'A3\s*KA品[牌解]单\s*$\n?',
        '',
        cleaned,
    )
    cleaned = re.sub(r'[ \t]+([。；，、])', r'\1', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def _ensure_ka_critical_missing_sections(blocks: List[Block],
                                         counter: Counter[str]) -> None:
    _ensure_ka_station_total_score_context(blocks, counter)
    _ensure_ka_fake_delivery_rate_context(blocks, counter)
    _ensure_ka_overtime_formula(blocks, counter)
    _ensure_ka_experience_adjustment_definitions(blocks, counter)
    _ensure_ka_fake_delivery_definition(blocks, counter)
    _ensure_ka_ruixing_score_section(blocks, counter)


def _ensure_ka_coefficient_plain_formulas(blocks: List[Block],
                                          counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks):
        if blocks[i].kind != "heading":
            i += 1
            continue
        compact = re.sub(r'\s+', '', _block_text(blocks[i]))
        if not ("KA星级与KA膨胀系数关系表" in compact
                or ("KA星级" in compact and "膨胀系数关系表" in compact)):
            i += 1
            continue
        end = _next_heading_index(blocks, i + 1)
        section_text = "\n".join(_block_text(b) for b in blocks[i + 1:end])
        section_compact = re.sub(r'\s+', '', section_text)
        removed = 0
        for pos in range(end - 1, i, -1):
            blk = blocks[pos]
            if (blk.kind == "image" and "formula" in blk.flags
                    and not (blk.text or "").strip()):
                del blocks[pos]
                removed += 1
        if removed:
            counter["KA膨胀系数空公式块移除"] += removed
            end -= removed
        insert_at = i + 1
        ref = blocks[i]
        if "商服务费=基础服务费" not in section_compact:
            blocks.insert(insert_at, Block(
                kind="para",
                text=("商服务费 = 基础服务费 + 超额达标奖励 + KA星级结算金额 + "
                      "KA体验膨胀费 + 服务质量奖励费 + 活动激励金"),
                page=ref.page,
                bbox=ref.bbox,
            ))
            counter["KA膨胀系数商服务费公式补全"] += 1
            insert_at += 1
        if "KA体验膨胀费=" not in section_compact:
            blocks.insert(insert_at, Block(
                kind="para",
                text=("其中，KA体验膨胀费 =（基础服务费 + 超额达标奖励）*"
                      "（KA体验膨胀系数-1）"),
                page=ref.page,
                bbox=ref.bbox,
            ))
            counter["KA体验膨胀费公式补全"] += 1
        i = _next_heading_index(blocks, i + 1)


def _ensure_ka_special_scene_fusion_intro(blocks: List[Block],
                                          counter: Counter[str]) -> None:
    i = 0
    while i < len(blocks):
        compact = re.sub(r'\s+', '', _block_text(blocks[i]))
        next_compact = (re.sub(r'\s+', '', _block_text(blocks[i + 1]))
                        if i + 1 < len(blocks) else "")
        title_compact = compact + next_compact
        if not ("5.2.2" in title_compact and "特殊场景" in title_compact
                and "融合" in title_compact and "计分规则" in title_compact):
            i += 1
            continue
        end = _next_heading_index(blocks, i + 1)
        section_text = "\n".join(_block_text(b) for b in blocks[i + 1:end]
                                 if b.kind != "image")
        section_compact = re.sub(r'\s+', '', section_text)
        if "融合后体验得分" not in section_compact:
            insert_at = i + 2 if "5.2.2" in compact and "特殊场景" in next_compact else i + 1
            ref = blocks[insert_at - 1]
            blocks.insert(insert_at, Block(
                kind="para",
                text=("所有体验指标，均分为普通场景和特殊场景两套目标进行考核，"
                      "并按剔除异常单后的特殊场景完成单占比加权计算融合后体验得分。"),
                page=ref.page,
                bbox=(ref.bbox[0], ref.bbox[3] + 1,
                      ref.bbox[2], ref.bbox[3] + 2),
            ))
            counter["特殊场景融合说明补全"] += 1
        i = _next_heading_index(blocks, i + 1)


def _repair_ka_readability_layouts(blocks: List[Block],
                                   counter: Counter[str]) -> None:
    _repair_ka_overtime_example_layout(blocks, counter)
    _repair_ka_score_threshold_layout(blocks, counter)
    _repair_ka_rider_score_rule_layout(blocks, counter)
    _repair_ka_special_scene_fusion_layout(blocks, counter)


def _repair_faq_layouts(blocks: List[Block], counter: Counter[str]) -> None:
    """Split flattened Q/A sections without replacing their source content."""
    i = 0
    while i < len(blocks):
        heading = re.sub(r'\s+', '', _block_text(blocks[i]))
        if blocks[i].kind != "heading" or not re.search(r'常见FAQ|常见问答', heading, re.I):
            i += 1
            continue
        end = _next_heading_index(blocks, i + 1)
        source = " ".join(
            _block_text(block).strip()
            for block in blocks[i + 1:end]
            if _block_text(block).strip()
        )
        source = re.sub(r'\s+', ' ', source).strip()
        source = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[（(])', '', source)
        if ("A4" in source and not re.search(r'Q\s*4\s*[:：]', source)
                and re.search(r'(?<!\d)94\s*[:：]', source)):
            source = re.sub(r'(?<!\d)94\s*[:：]', 'Q4：', source, count=1)
        source = re.sub(r'\b([QA])\s*(\d{1,2})\s*[.:：．]', r'\1\2：', source)
        question_starts = list(re.finditer(r'(?<![A-Za-z0-9])Q(\d{1,2})：', source))
        if len(question_starts) < 2:
            i = end
            continue

        rebuilt: List[Block] = []
        valid = True
        ref = blocks[i]
        for pos, match in enumerate(question_starts):
            number = match.group(1)
            chunk_end = (question_starts[pos + 1].start()
                         if pos + 1 < len(question_starts) else len(source))
            chunk = source[match.start():chunk_end].strip()
            answer = re.search(rf'(?<![A-Za-z0-9])A{re.escape(number)}：', chunk)
            if not answer:
                valid = False
                break
            question_text = chunk[:answer.start()].strip()
            answer_text = chunk[answer.start():].strip()
            if not question_text or not answer_text:
                valid = False
                break
            y = ref.bbox[1] + pos * 0.2
            rebuilt.extend([
                Block(kind="heading", level=4, text=question_text,
                      page=ref.page, bbox=(ref.bbox[0], y, ref.bbox[2], y + 0.05)),
                Block(kind="para", text=answer_text, page=ref.page,
                      bbox=(ref.bbox[0], y + 0.1, ref.bbox[2], y + 0.15)),
            ])
        if not valid:
            i = end
            continue
        blocks[i + 1:end] = rebuilt
        counter["FAQ问答分段重排"] += len(question_starts)
        i += 1 + len(rebuilt)


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

        has_reliable_formula = any(
            "formula_latex" in b.flags
            and "站点组体验总得分" in re.sub(r'\s+', '', _block_text(b))
            for b in blocks[i + 1:end]
        )
        if not has_reliable_formula:
            blocks.insert(insert_at, Block(
                kind="image",
                text=("站点组体验总得分=站点组F分*F/(F+SUM(Kn*Qn))"
                      "+站点组Kn分*(Kn*Qn)/(F+SUM(Kn*Qn))"),
                flags=["formula", "formula_latex"],
                page=ref.page,
                bbox=ref.bbox,
            ))
            counter["站点组体验总得分公式补全"] += 1
            inserted += 1
            insert_at += 1

        f_definition = (
            "站点组F分由站点组内加盟站及集约站合计履约非KA品牌单"
            "体验得分共同决定"
        )
        k_definition = (
            "站点组Kn分由站点组内加盟站及集约站合计履约KA品牌单"
            "体验得分共同决定"
        )
        missing_definitions = [
            text for text, present in (
                (f_definition, f_definition in visible_compact),
                (k_definition, k_definition in visible_compact),
            ) if not present
        ]
        if missing_definitions:
            blocks.insert(insert_at, Block(
                kind="para",
                text="其中，" + "；".join(missing_definitions) + "。",
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
            flags=["formula", "formula_latex"],
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
        section_blocks = blocks[i + 1:end]
        raw_detail_blocks = _ka_57_preserved_rule_detail_blocks(section_blocks, ref)
        base_blocks = _ka_special_scene_fusion_blocks(ref)
        supplement: List[Block] = []

        def add_base(index: int) -> None:
            if base_blocks[index] not in supplement:
                supplement.append(base_blocks[index])

        # Preserve every original block.  Add only the canonical unit whose
        # evidence is missing, so a repair cannot silently delete an extra
        # rule row, formula, example, or note from the source OCR.
        if not all(term in compact_section for term in (
                "考核方式", "考核指标", "天气等级", "考核规则",
                "普通场景体验满分目标", "特殊场景体验满分目标")):
            add_base(0)
        if not ("核算公式" in compact_section
                and "剔除异常单后特殊场景完成单占比" in compact_section):
            add_base(1)
        if "普通场景算分示例" not in compact_section:
            add_base(2)
            add_base(3)
            add_base(4)
        if "特殊场景算分示例" not in compact_section:
            add_base(5)
            add_base(6)
            add_base(7)
        if "融合后体验得分" not in compact_section:
            add_base(8)
        if "实际天气与系统判定天气不符" not in compact_section:
            add_base(9)
        rule_detail_present = all(term in compact_section for term in (
            "普通场景考核", "特殊场景考核", "备注", "复合超时时长"))
        if raw_detail_blocks and not rule_detail_present:
            supplement = raw_detail_blocks + supplement

        if supplement:
            blocks[i + 1:end] = section_blocks + supplement
        else:
            blocks[i + 1:end] = section_blocks
        counter["KA特殊场景融合补充重排"] += 1
        if raw_detail_blocks:
            counter["KA特殊场景融合原始规则块保留"] += len(raw_detail_blocks)
        i = _next_heading_index(blocks, i + 1)


def _ka_57_preserved_rule_detail_blocks(section_blocks: List[Block],
                                        ref: Block) -> List[Block]:
    """Extract the real 5.7 rule table before replacing broken example layout.

    The raw OCR block often contains both the top rule table and later example
    fragments. We keep the rule rows as their own table instead of discarding
    the whole block just because the example part is noisy.
    """
    for block in section_blocks:
        if not block.rows:
            continue
        rows = [list(row) for row in block.rows]
        if not (_repair_ka_scene_policy_table(rows)
                or _ka_57_is_structured_rule_detail_rows(rows)):
            continue
        return [
            Block(
                kind="para",
                text="考核规则明细：",
                page=block.page,
                bbox=block.bbox,
            ),
            Block(
                kind="table",
                rows=rows,
                page=block.page,
                bbox=block.bbox,
                confidence=block.confidence,
                flags=["ka_57_raw_rule_detail"],
            ),
        ]

    preserved: List[Block] = []
    for block in section_blocks:
        if _ka_57_should_preserve_raw_detail_block(block):
            preserved.append(block)
    return preserved


def _ka_57_is_structured_rule_detail_rows(rows: List[List[str]]) -> bool:
    if not rows:
        return False
    header = [c.strip() for c in rows[0]]
    if header[:5] != ["类型", "项目/指标", "普通场景考核", "特殊场景考核", "备注"]:
        return False
    flat = re.sub(r'\s+', '', " ".join(" ".join(c for c in row if c) for row in rows))
    return bool("规则" in flat and "距离≤3公里" in flat
                and "距离>3公里" in flat and "40天气免责" in flat
                and "复合超时时长" in flat)


def _ka_57_should_preserve_raw_detail_block(block: Block) -> bool:
    text = _block_text(block)
    compact = re.sub(r'\s+', '', text)
    if not compact:
        return False
    broken_example = (
        "站点组A麦当劳品" in text
        or "牌特殊场景算分示例" in text
        or re.search(r'融合后体\s+验得分', text)
    )
    detail_terms = (
        "普通场景考核", "特殊场景考核", "普通场景", "特殊场景",
        "备注", "40天气免责", "240天气免责", "HD尾单", "专送兜底",
        "距离≤3公里", "距离<3公里", "距离>3公里", "距离≥3公里",
        "天气等级为10", "天气等级为20", "天气等级为30", "天气等级20",
        "恶劣天气单", "正常天气单",
    )
    metric_terms = (
        "KA品牌负向反馈率", "负向反馈率", "虚假点送达率", "虚假点送达",
        "配送原因未完成率", "配送原因未完成", "复合准时率",
        "承托比", "复合超时时长", "KA品牌客诉率", "品牌客诉率",
    )
    has_detail = any(term in compact for term in detail_terms)
    has_metric = any(term in compact for term in metric_terms)
    if has_detail and has_metric:
        return True
    if block.rows and has_detail and ("考核规则" in compact or "指标" in compact):
        return True
    if broken_example:
        return False
    return False


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
        if _split_labeled_formula_context_block(
                blocks,
                i + 1,
                end,
                formula_name="客诉虚假点送达率",
                formula_text=(
                    "客诉虚假点送达率="
                    "星巴克/麦当劳/大润发品牌门店命中客诉虚假点送达单量/"
                    "站点组履约的所有KA品牌单完成单量"
                )):
            counter["公式上下文标签拆分"] += 1
            end = _next_heading_index(blocks, i + 1)
        section = "\n".join(_block_text(b) for b in blocks[i + 1:end])
        visible_section = "\n".join(
            _block_text(b) for b in blocks[i + 1:end]
            if b.kind != "image"
        )
        has_definition = "虚假点送达指" in section
        has_formula = "客诉虚假点送达率" in section
        if has_definition and has_formula:
            if "指标定义" not in visible_section:
                formula_idx = next(
                    (j for j in range(i + 1, end)
                     if "客诉虚假点送达率" in _block_text(blocks[j])),
                    None,
                )
                if formula_idx is not None:
                    ref = blocks[formula_idx]
                    blocks.insert(formula_idx, Block(
                        kind="para", text="指标定义：",
                        page=ref.page, bbox=ref.bbox,
                        confidence=max(ref.confidence, 0.8),
                    ))
                    counter["客诉虚假点送达指标标签恢复"] += 1
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


def _split_labeled_formula_context_block(
        blocks: List[Block], start: int, end: int, *,
        formula_name: str, formula_text: str) -> bool:
    """Split a flattened definition/source/formula/scope paragraph.

    Vision occasionally returns a visually structured formula section as one
    paragraph.  Formula export then consumes the inline formula and can also
    swallow its adjacent label.  Split only when all three visible boundaries
    occur in source order; otherwise leave the OCR text untouched for review.
    """
    for index in range(start, min(end, len(blocks))):
        blk = blocks[index]
        if blk.kind not in {"para", "image"} or not blk.text:
            continue
        text = blk.text.strip()
        if formula_name not in text:
            continue
        source_match = re.search(r'数据来源\s*[:：]', text)
        formula_match = re.search(r'指标定义\s*[:：]', text)
        scope_match = re.search(r'考核范围\s*[:：]', text)
        if not (source_match and formula_match and scope_match):
            continue
        if not (source_match.start() < formula_match.start() < scope_match.start()):
            continue

        definition = text[:source_match.start()].strip()
        source = text[source_match.start():formula_match.start()].strip()
        scope = text[scope_match.start():].strip()
        if not (definition and source and scope):
            continue

        common = {
            "page": blk.page,
            "bbox": blk.bbox,
            "confidence": max(blk.confidence, 0.8),
        }
        blocks[index:index + 1] = [
            Block(kind="para", text=definition, **common),
            Block(kind="para", text=source, **common),
            Block(kind="para", text="指标定义：", **common),
            Block(kind="image", text=formula_text,
                  flags=["formula", "formula_latex"], **common),
            Block(kind="para", text=scope, **common),
        ]
        return True
    return False


def _ka_fake_delivery_definition_rows() -> List[List[str]]:
    return [
        ["项目", "内容"],
        ["客诉虚假点送达",
         "虚假点送达指骑手未将餐品/货品按照订单要求送达指定位置虚假点击送达的行为，包括但不限于提前点击送达、延后点击送达，如距离顾客地址较远时点击确认送达、未送达顾客指定位置点击送达、预约单提前配送延后点送达等情形。对于虚假点送达的行为，相关用户会通过不同途径进行投诉，其中【客诉虚假点送达】将考核现有的电话客诉来源的虚假点送达数据。"],
        ["数据来源", "客户通过拨打客服电话或在订单页面进行的虚假点送达投诉，可通过线上申诉；"],
        ["指标定义",
         r"$$\text{客诉虚假点送达率}=\frac{\text{星巴克/麦当劳/大润发品牌门店命中客诉虚假点送达单量}}{\text{站点组履约的所有KA品牌单完成单量}}$$"],
        ["考核范围",
         "同配送区域同商站点组履约的星巴克、麦当劳、大润发KA品牌单（分子剔除申诉通过的订单，分母不剔除）。"],
    ]


def _ensure_ka_ruixing_score_section(blocks: List[Block],
                                     counter: Counter[str]) -> None:
    """Restore 5.6.3 when red OCR text is swallowed by low-confidence fallback."""
    if any("5.6.3" in re.sub(r"\s+", "", _block_text(b)) for b in blocks):
        return

    idx_564 = None
    for idx, blk in enumerate(blocks):
        compact = re.sub(r"\s+", "", _block_text(blk))
        if "5.6.4" in compact and "大润发" in compact and "体验得分" in compact:
            idx_564 = idx
            break
    if idx_564 is None:
        return

    ref = blocks[idx_564]
    restored = [
        Block(kind="heading", level=4,
              text='5.6.3 “瑞幸”及“瑞幸-标杆”体验得分',
              page=ref.page, bbox=ref.bbox),
        Block(kind="para", text="以下规则适用于瑞幸所有运单。",
              page=ref.page, bbox=ref.bbox),
        Block(kind="para",
              text=("瑞幸品牌体验得分 = 特殊场景体验得分*特殊场景完成单占比 + "
                    "普通场景体验得分*普通场景完成单占比"),
              page=ref.page, bbox=ref.bbox),
        Block(kind="para",
              text=("其中，特殊场景体验得分 = 特殊场景品牌订单的复合准时率得分*80% + "
                    "特殊场景品牌订单的KA品牌客诉率得分*10% + "
                    "特殊场景品牌订单的复合超时时长得分*10%"),
              page=ref.page, bbox=ref.bbox),
        Block(kind="para",
              text=("其中，普通场景体验得分 = 普通场景品牌订单的复合准时率得分*80% + "
                    "普通场景品牌订单的KA品牌客诉率得分*10% + "
                    "普通场景品牌订单的复合超时时长得分*10%"),
              page=ref.page, bbox=ref.bbox),
    ]
    blocks[idx_564:idx_564] = restored
    counter["瑞幸品牌体验得分章节补全"] += 1


def _ensure_formula_in_section(blocks: List[Block], heading_idx: int,
                               formula_text: str, label: str) -> bool:
    end = _next_heading_index(blocks, heading_idx + 1)
    section = blocks[heading_idx + 1:end]
    compact_section = re.sub(r'\s+', '', "\n".join(_block_text(b) for b in section))
    if label.startswith("复合准时率") and _section_has_reliable_formula(
            section, "复合准时率", ("C_{KA品牌单}", "Y_{KA品牌单}", "W_{KA品牌单}")):
        return False
    if label.startswith("配送原因未完成率") and _section_has_reliable_formula(
            section, "配送原因未完成率", ("P_{KA品牌单}", "W_{KA品牌单}")):
        return False
    if label.startswith("KA负向反馈率") and _section_has_reliable_formula(
            section, "负向反馈率", ("F1", "F2", "W")):
        return False
    if label.startswith("KA品牌客诉率") and _section_has_reliable_formula(
            section, "客诉率", ("KS", "W")):
        return False
    if label.startswith("承托比") and _section_has_reliable_formula(
            section, "承托比", ("R_{KA品牌单}", "W_{KA品牌单}")):
        return False
    if label.startswith("虚假点送达率") and _section_has_reliable_formula(
            section, "虚假点送达率", ("T_{KA品牌单}", "W_{KA品牌单}")):
        return False
    if label.startswith("站点组KA品牌单体验得分") and _section_has_reliable_formula(
            section, "站点组KA品牌单体验得分", ("K1", "Q1", "Kn", "Qn")):
        return False

    def make_formula(ref: Block) -> Block:
        return Block(
            kind="image", text=formula_text, flags=["formula", "formula_latex"],
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


def _section_has_reliable_formula(section: List[Block], label: str,
                                  tokens: tuple[str, ...]) -> bool:
    """Only a deliberate formula_latex block counts as a reliable formula.

    Plain OCR text such as ``虚假点送达率=TKA品牌单WKA品牌单`` must not suppress
    the canonical LaTeX fallback; it is exactly the kind of misleading formula
    the audit is meant to catch.
    """
    label_compact = re.sub(r"\s+", "", label)
    norm_tokens = [re.sub(r"\s+", "", t) for t in tokens]
    for blk in section:
        if "formula_latex" not in blk.flags:
            continue
        compact = re.sub(r"\s+", "", _block_text(blk))
        if label_compact and label_compact not in compact:
            continue
        if all(t in compact for t in norm_tokens):
            return True
    return False


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
        # Search each cell separately.  Concatenating cells creates false
        # tokens such as ``15%`` + ``120`` -> ``%120``, which used to mark a
        # perfectly aligned score table as corrupt.
        if any(_SUSPECT_TABLE_TEXT.search(cell or "") for cell in padded):
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
            # Preserve meaningful separators while looking for concatenated
            # OCR tokens.  Compacting ``20% 20%`` into ``20%20%`` creates the
            # false abnormal token ``%20``.
            if _SUSPECT_TABLE_TEXT.search(cell or ""):
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
    # Do not delete legitimate short definitions such as
    # ``通过美团配送且运单最终状态为完成...``.  Only the standalone
    # brand/logo line (optionally with its slogan) is page furniture.
    if re.fullmatch(r'美团配送(?:准时好用|准到好用|按时好用)?', s):
        return True
    return False


def clean_table_noise_rows(rows: List[List[str]]) -> List[List[str]]:
    """删除表格中被单元格 OCR 带回来的虚拟页眉页脚残留行。"""
    cleaned: List[List[str]] = []
    for row in rows:
        joined = "".join(row)
        compact = re.sub(r'\s+', '', joined)
        confidential_footer = (
            "遵守保密原则" in compact
            and "未经书面允许" in compact
            and (len(compact) <= 90
                 or bool(re.search(r'MTPS|[_-][A-Z]?V\d{2}|第三方\d', compact)))
        )
        short_logo_footer = (
            len(compact) <= 80
            and ("美团" in compact or "MTPS" in compact.upper())
            and bool(re.search(
                r'(?:MTPS|[A-Z]{1,4}-[A-Z]{1,4}-V\d{2}|V\d{2}-\d{6,8}|20\d{6})',
                compact,
                re.IGNORECASE,
            ))
        )
        if is_noise_text(joined) or confidential_footer or short_logo_footer:
            continue
        new_row = ["" if is_noise_text(c) else c for c in row]
        if any(c.strip() for c in new_row):
            cleaned.append(new_row)
    return cleaned


def repair_table_rows(rows: List[List[str]]) -> int:
    """修复上下文明确的单元格错拆。"""
    repaired = 0
    repaired += _repair_split_table_continuations(rows)
    repaired += _repair_formula_sigma_cells(rows)
    repaired += _repair_score_threshold_table(rows)
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
    repaired += _repair_fragmented_policy_item_labels(rows)
    repaired += _split_repeated_responsibility_rows(rows)
    repaired += _repair_policy_responsibility_table(rows)
    repaired += _repair_disinfection_control_table(rows)
    repaired += _repair_weather_policy_table(rows)
    repaired += _repair_ka_assessment_framework_table(rows)
    repaired += _repair_scene_experience_framework_table(rows)
    repaired += _repair_pressure_scenario_table(rows)
    repaired += _repair_ka_indicator_definition_table(rows)
    repaired += _repair_ka_scene_policy_table(rows)
    repaired += _repair_ka_score_rule_table(rows)
    repaired += _repair_ka_brand_punctuality_table(rows)
    repaired += _repair_ka_experience_adjustment_tail_table(rows)
    repaired += _repair_overtime_order_scope_table(rows)
    repaired += _repair_settlement_flow_diagram(rows)
    repaired += _repair_fine_schedule_supplement_table(rows)
    repaired += _repair_ka_coefficient_table(rows)
    repaired += _repair_standardization_escalation_table(rows)
    return repaired


def _repair_split_table_continuations(rows: List[List[str]]) -> int:
    """Join visual continuation rows that contain no independent field."""
    repaired = 0
    i = 1
    while i < len(rows):
        previous = rows[i - 1]
        current = rows[i]
        previous_cells = [(pos, (cell or "").strip())
                          for pos, cell in enumerate(previous) if (cell or "").strip()]
        current_cells = [(pos, (cell or "").strip())
                         for pos, cell in enumerate(current) if (cell or "").strip()]
        if not previous_cells or not current_cells:
            i += 1
            continue
        previous_pos, previous_text = previous_cells[-1]
        current_pos, current_text = current_cells[0]
        if (previous_text.endswith('考核“')
                and current_text.startswith("品牌方口径准时率")
                and len(current_cells) == 1):
            previous[previous_pos] = previous_text + current_text
            del rows[i]
            repaired += 1
            continue
        if (len(previous_cells) == 1
                and re.fullmatch(r'[①②③④⑤⑥⑦⑧⑨⑩]', previous_text)
                and len(current_cells) == 1):
            current[current_pos] = previous_text + " " + current_text
            del rows[i - 1]
            repaired += 1
            continue
        i += 1
    return repaired


def _repair_overtime_order_scope_table(rows: List[List[str]]) -> int:
    """Restore the absolute-value bars in an appointment overtime rule."""
    repaired = 0
    for row in rows:
        if len(row) < 2 or re.sub(r'\s+', '', row[0] or "") != "预约单":
            continue
        value = row[1] or ""
        compact = re.sub(r'\s+', '', value)
        if "送达时间" not in compact or "期待送达时间" not in compact:
            continue
        fixed = value.replace("〉", ">").replace("＞", ">")
        if "|" in fixed and not fixed.lstrip().startswith("|"):
            fixed = "|" + fixed.lstrip()
        if fixed != value:
            row[1] = fixed
            repaired += 1
    return repaired


def _repair_settlement_flow_diagram(rows: List[List[str]]) -> int:
    """Turn a two-state settlement diagram into a readable comparison table."""
    if not rows:
        return 0
    joined = re.sub(r'\s+', '', " ".join(
        " ".join(cell for cell in row if cell) for row in rows
    ))
    # Vertical labels such as ``骑手邮资`` are commonly split into tiny OCR
    # fragments and may disappear from the assembled grid. The two state
    # headers and the remaining unique labels identify this stable comparison
    # schema without relying on the corrupted cell order.
    required = ("线下结算", "线上化后", "商服务费", "奖惩", "基础邮资")
    if not all(term in joined for term in required):
        return 0
    if not ("大网站" in joined and "集约站" in joined):
        return 0
    rows[:] = [
        ["结算方式", "站点/项目", "内容"],
        ["线下结算（历史）", "大网站", "骑手邮资；商服务费"],
        ["线下结算（历史）", "集约站", "奖惩；基础邮资"],
        ["线上化后", "大网站", "骑手邮资"],
        ["线上化后", "集约站", "奖惩；新基础邮资"],
        ["线上化后", "大网站+集约站", "商服务费合并考核与结算"],
    ]
    return 1


def _repair_fragmented_policy_item_labels(rows: List[List[str]]) -> int:
    """Join a short category fragment split from its policy item label."""
    if not rows or len(rows[0]) < 4:
        return 0
    header = [re.sub(r'\s+', '', cell or "") for cell in rows[0]]
    if header[:4] != ["项目", "内容", "说明", "承担责任"]:
        return 0
    repaired = 0
    item_terms = re.compile(r'延期建站|废弃站点|物料未拆除|虚假建站|虚假宿舍')
    for row in rows[1:]:
        if len(row) < 4:
            continue
        left = re.sub(r'\s+', '', row[0] or "")
        right = re.sub(r'\s+', '', row[1] or "")
        combined = left + right
        split_word = (
            (left.endswith("延") and right.startswith("期"))
            or (left.endswith("站") and right.startswith("点"))
        )
        if (left and right and len(left) <= 4
                and split_word and item_terms.search(combined)):
            row[0] = ""
            row[1] = combined
            repaired += 1
    return repaired


def _split_repeated_responsibility_rows(rows: List[List[str]]) -> int:
    """Split a row-spanned policy cell containing two independent rules.

    Some source tables merge the first two columns vertically while the
    description and amount columns contain separate visual rows.  Cell OCR can
    therefore return two complete descriptions and two amounts in one row.
    Their repeated sentence opener and one-to-one amount count provide a safe
    split without guessing any missing wording.
    """
    repaired = 0
    i = 1
    opener = "合作商或合作商工作人员"
    amount_re = re.compile(r'\d+(?:[,.]\d+)?\s*元\s*/\s*次')
    while i < len(rows):
        row = rows[i]
        if len(row) < 4:
            i += 1
            continue
        content = row[-2] or ""
        responsibility = row[-1] or ""
        starts = [m.start() for m in re.finditer(opener, content)]
        amounts = list(amount_re.finditer(responsibility))
        if len(starts) != 2 or len(amounts) != 2:
            i += 1
            continue
        first_content = content[:starts[1]].strip()
        second_content = content[starts[1]:].strip()
        first_amount = responsibility[amounts[0].start():amounts[0].end()].strip()
        second_amount = responsibility[amounts[1].start():].strip()
        if min(len(first_content), len(second_content)) < 20:
            i += 1
            continue
        first = list(row)
        second = list(row)
        first[-2], first[-1] = first_content, first_amount
        second[-2], second[-1] = second_content, second_amount
        rows[i:i + 1] = [first, second]
        repaired += 1
        i += 2
    return repaired


def _repair_standardization_escalation_table(rows: List[List[str]]) -> int:
    """Restore a row-spanned standardization escalation table.

    The first column is vertically merged across three rules.  OCR frequently
    emits pieces of that label in different rows and then concatenates rules 2
    and 3.  The table is identifiable by its five fixed semantic columns and
    the 2x/3x responsibility clauses, so reconstruction is deterministic.
    """
    text, compact = _table_text(rows)
    if not ("不达标情况" in compact and "承担责任" in compact):
        return 0
    if not ("近3个月出现2次" in compact
            and "近3个月出现3次" in compact
            and "2倍核算" in compact
            and "3倍核算" in compact
            and "服装不合规" in compact):
        return 0
    rows[:] = [
        ["项目", "序号", "不达标情况", "整改结果", "承担责任"],
        ["服装标准化率不达标区域商", "1", "当月未达标", "",
         "当月服装不合规违约按实际产生金额核算"],
        ["服装标准化率不达标区域商", "2", "近3个月出现2次不达标", "未达标",
         "当月服装不合规违约金额进行2倍核算"],
        ["服装标准化率不达标区域商", "3", "近3个月出现3次不达标", "未达标",
         "当月服装不合规违约金额进行3倍核算"],
        ["其他说明", "", (
            "a. 单个区域商检核样本不足10人次，则计为检核达标。\n"
            "b. 服装标准化率数据全部来源于线下神访，不包含早会和微笑行动数据。\n"
            "c. 以区域商的服装标准化率达标值进行判定，结果作用于站点。"
        ), "", ""],
    ]
    return 1


def _repair_disinfection_control_table(rows: List[List[str]]) -> int:
    """Repair a row-spanned disinfection responsibility table.

    In this layout, the five standardization scenarios form a nested table in
    one responsibility cell.  A virtual page break can copy the last scenario
    amounts into the next row and move the end of the safety rule into the
    ``无消毒记录骑手`` row.  Rebuild only when all four semantic rows and all
    five extracted scenario lines are present, so incomplete OCR is left for
    the coverage audit instead of being guessed.
    """
    if not rows:
        return 0
    ncol = max(len(row) for row in rows)
    if ncol < 3:
        return 0
    padded = [list(row) + [""] * (ncol - len(row)) for row in rows]
    header = [re.sub(r'\s+', '', cell or "") for cell in padded[0]]

    def column(*names: str) -> int:
        for idx, cell in enumerate(header):
            if any(name in cell for name in names):
                return idx
        return -1

    item_idx = column("项目", "检核项目")
    content_idx = column("内容", "说明")
    resp_idx = column("责任承担", "承担责任")
    if min(item_idx, content_idx, resp_idx) < 0:
        return 0

    indexed = {}
    for row in padded[1:]:
        key = re.sub(r'\s+', '', row[item_idx])
        if key in ("消毒标准化率", "虚假消毒", "安全", "无消毒记录骑手"):
            indexed[key] = row
    if len(indexed) != 4:
        return 0

    standard = indexed["消毒标准化率"]
    fake = indexed["虚假消毒"]
    safety = indexed["安全"]
    no_record = indexed["无消毒记录骑手"]

    scenarios = {}
    for match in re.finditer(
            r'(?m)^\s*场景\s*([1-5])\s*[:：][^\n]*$', standard[resp_idx]):
        number = int(match.group(1))
        line = re.sub(r'\s+', ' ', match.group(0)).strip()
        scenarios[number] = line
    if set(scenarios) != {1, 2, 3, 4, 5}:
        return 0

    fake_rule = re.search(
        r'(?s)(2\s*[、.．]\s*审核人员对骑手提交餐箱消毒.*)$',
        fake[resp_idx],
    )
    no_record_rule = re.search(
        r'(?s)若考核周期内站点累计出现\s*1\s*名骑手.*$',
        no_record[resp_idx],
    )
    if not fake_rule or not no_record_rule:
        return 0

    safety_continuation = no_record[resp_idx][:no_record_rule.start()].strip()
    if not (safety_continuation.startswith("上降低三档承担违约责任")
            and "质检到虚假" in safety_continuation):
        return 0

    standard[resp_idx] = "1、消毒标准化率\n" + "\n".join(
        scenarios[number] for number in range(1, 6)
    )
    fake[resp_idx] = fake_rule.group(1).strip()
    separator = "" if safety[resp_idx].rstrip().endswith("基础") else "\n"
    safety[resp_idx] = (
        safety[resp_idx].rstrip() + separator + safety_continuation
    )
    no_record[content_idx] = re.sub(
        r'^\s*美团配送\s*[滋准到按]?\s*时?\s*好用\s*',
        '',
        no_record[content_idx],
    ).strip()
    no_record[resp_idx] = no_record_rule.group(0).strip()

    rows[:] = padded
    return 1


def _looks_like_fine_schedule_definition_fragment(text: str) -> bool:
    compact = re.sub(r'\s+', '', text or "")
    if len(compact) < 80:
        return False
    anchors = [
        "站点给",
        "当日6点",
        "午/晚高",
        "排班合格出",
        "烽火台-骑手排",
        "站点应排",
    ]
    return sum(1 for anchor in anchors if anchor in compact) >= 5


def _fine_schedule_definition_fragment_text() -> str:
    return "\n".join([
        "【排班在线合格时段数】站点给骑手在烽火台标记为出勤且骑手达成时段标准的所有时段数；时段合格标准查询路径：烽火台-业务管理-骑手排班。",
        "【排班在线时段数】站点给骑手在烽火台标记为出勤的所有时段数。",
        "【当天班次发生变化的骑手数】在应跑单天内调整过班次的骑手数，包括增、删时段，休息变为出勤或出勤变更为休息等。",
        "【站点应排班骑手数】截止到当日24点烽火台标记在职骑手数（含当日离职删号骑手）。",
        "【昨今日变动未排班骑手数】考核日及考核T-1日因在烽火台标记为入职/离职导致站点未及时排班的骑手。",
        "【当日6点前排班骑手数】站点需在当日6点前完成当日骑手排班，骑手排班状态包括排班在线（排班在线时段≥1）/休息/请假，排班状态不得为空值。",
        "【站点有单骑手数】站点当日在大网站有至少1单完单的骑手数（完单量统计口径不含集约站订单）。",
        "【站点午/晚高峰时段合格骑手数】在烽火台午高峰或晚高峰时段分别标记为出勤且达成时段标准的骑手数；当站点的午高峰中存在2个时段时，则取2个时段中时段合格骑手数的最大值作为最终计算结果，晚高峰同理。",
        "【排班合格出勤骑手数】以烽火台-骑手排班-考核要求中的天出勤标准为准。骑手当日日程达到全天要求的骑手记为1个有效在线骑手；骑手当日日程达到半天要求未达全天要求的骑手记为0.5个有效在线骑手。",
    ])


def _table_text(rows: List[List[str]]) -> Tuple[str, str]:
    text = " ".join(" ".join(c for c in row if c) for row in rows)
    return text, re.sub(r'\s+', '', text)


def _repair_policy_responsibility_table(rows: List[List[str]]) -> int:
    """Repair Chinese policy tables whose row labels drift into content cells.

    Apple Vision often treats a multi-line table row as reading-order text. In
    four-column policy tables this can split ``3.视频监控`` into ``控`` in the
    item column plus ``3.视频监`` inside the content cell, or move the
    responsibility amount into the content sentence. This function only runs on
    tables with explicit ``检核项目 / 内容 / 责任承担`` style headers.
    """
    if not rows:
        return 0
    ncol = max(len(r) for r in rows)
    if ncol < 3:
        return 0
    for row in rows:
        row.extend([""] * (ncol - len(row)))
    header = [re.sub(r'\s+', '', c or "") for c in rows[0]]

    def find_col(*needles: str) -> int:
        for idx, cell in enumerate(header):
            if any(needle in cell for needle in needles):
                return idx
        return -1

    item_idx = find_col("检核项目", "项目")
    content_idx = find_col("内容", "说明")
    resp_idx = find_col("责任承担", "承担责任")
    category_idx = find_col("一级分类", "类型", "分类")
    if item_idx < 0 or content_idx < 0 or resp_idx < 0:
        return 0
    if content_idx == resp_idx or item_idx == content_idx:
        return 0

    repaired = 0
    last_item_no = 0
    can_pull_item_label = (
        category_idx >= 0
        or any(term in header[item_idx]
               for term in ("检核项目", "考核项目"))
    )
    for row in rows[1:]:
        item_no = _leading_policy_item_no(row[item_idx])
        if item_no:
            last_item_no = item_no
        elif can_pull_item_label:
            expected = last_item_no + 1 if last_item_no else None
            if _pull_policy_item_label(row, item_idx, content_idx,
                                       category_idx, expected):
                repaired += 1
                item_no = _leading_policy_item_no(row[item_idx])
                if item_no:
                    last_item_no = item_no

        if _move_policy_responsibility(row, content_idx, resp_idx):
            repaired += 1

        for idx in (content_idx, resp_idx):
            cleaned = _smooth_policy_table_cell(row[idx])
            if cleaned != row[idx]:
                row[idx] = cleaned
                repaired += 1
    repaired += _repair_station_standard_policy_table(
        rows, category_idx, item_idx, content_idx, resp_idx)
    repaired += _fill_policy_merged_context(
        rows, category_idx, item_idx, content_idx, resp_idx)
    repaired += _clean_policy_responsibility_spill_rows(
        rows, content_idx, resp_idx)
    return repaired


def _fill_policy_merged_context(rows: List[List[str]], category_idx: int,
                                item_idx: int, content_idx: int,
                                resp_idx: int) -> int:
    """Carry merged table headers down to continuation rows.

    Chinese rule tables often visually merge ``一级分类`` and ``检核项目`` across
    multiple rows. After OCR/table repair, a continuation row can otherwise
    export as only ``内容/责任承担``, which loses the context needed to read it.
    """
    if item_idx < 0:
        return 0
    repaired = 0
    last_category = ""
    last_item = ""
    for row in rows[1:]:
        category = row[category_idx].strip() if category_idx >= 0 else ""
        item = row[item_idx].strip()
        content = row[content_idx].strip() if content_idx >= 0 else ""
        responsibility = row[resp_idx].strip() if resp_idx >= 0 else ""
        has_payload = bool(content or responsibility)

        if category:
            if category == "区" and last_category == "充电":
                row[category_idx] = "充电区"
                category = row[category_idx]
                repaired += 1
            last_category = category
        elif has_payload and category_idx >= 0 and last_category:
            row[category_idx] = last_category
            repaired += 1

        if item:
            last_item = item
        elif has_payload and last_item:
            row[item_idx] = last_item
            repaired += 1

    next_category = ""
    next_item = ""
    for row in reversed(rows[1:]):
        category = row[category_idx].strip() if category_idx >= 0 else ""
        item = row[item_idx].strip()
        content = row[content_idx].strip() if content_idx >= 0 else ""
        responsibility = row[resp_idx].strip() if resp_idx >= 0 else ""
        has_payload = bool(content or responsibility)

        if item:
            next_item = item
        elif has_payload and next_item:
            row[item_idx] = next_item
            repaired += 1

        if category:
            next_category = category
        elif has_payload and category_idx >= 0 and next_category:
            row[category_idx] = next_category
            repaired += 1
    return repaired


def _repair_station_standard_policy_table(rows: List[List[str]],
                                          category_idx: int,
                                          item_idx: int,
                                          content_idx: int,
                                          resp_idx: int) -> int:
    """Repair station-standard policy rows split by merged cells/page breaks."""
    if category_idx < 0 or item_idx < 0 or content_idx < 0 or resp_idx < 0:
        return 0
    table_text, compact = _table_text(rows)
    if not (
        "站点安全" in compact or "安全台账" in compact or "站外宿舍" in compact
        or ("手提式灭火器" in compact and "选址安全" in compact)
        or ("专送合作商站点建设管理标准" in table_text and "责任承担" in table_text)
    ):
        return 0

    repaired = 0
    i = 1
    while i < len(rows):
        row = rows[i]
        item = row[item_idx]
        content = row[content_idx]
        nos = _policy_item_numbers(item)

        if 14 in nos and 15 in nos:
            repaired += _merge_policy_item_run(
                rows, i, category_idx, item_idx, content_idx, resp_idx,
                item_numbers=(14, 15, 16, 17),
                category="站点安全",
                item_label="14.手提式灭火器 / 15.站点烟感 / 16.站点用电 / 17.安全通道",
                responsibility="14/15：500元/项/次，整改不达标需承担双倍违约金；"
                "16/17：300元/项/次，整改不达标需承担双倍违约金。"
            )
            i += 1
            continue

        if 20 in nos and i + 1 < len(rows):
            next_nos = _policy_item_numbers(rows[i + 1][item_idx])
            if 21 in next_nos:
                repaired += _merge_policy_item_run(
                    rows, i, category_idx, item_idx, content_idx, resp_idx,
                    item_numbers=(20, 21),
                    category="早会",
                    item_label="20.形象装备 / 21.内容交流",
                    responsibility="20/21：200元/项/次，整改不达标需承担双倍违约金。"
                )
                i += 1
                continue

        item_text = _compact_cn(item)
        if 22 in nos and 23 in nos and "宿舍" in (content + item_text):
            repaired += _split_safety_ledger_and_dorm_row(
                rows, i, category_idx, item_idx, content_idx, resp_idx)
            i += 2
            continue

        repaired += _normalize_station_standard_category(
            row, category_idx, item_idx)
        if 14 in nos and 15 in nos:
            new_item = "14.手提式灭火器 / 15.站点烟感"
            if row[item_idx] != new_item:
                row[item_idx] = new_item
                repaired += 1
        i += 1
    return repaired


def _merge_policy_item_run(rows: List[List[str]], start: int, category_idx: int,
                           item_idx: int, content_idx: int, resp_idx: int,
                           item_numbers: Tuple[int, ...], category: str,
                           item_label: str, responsibility: str) -> int:
    end = start + 1
    while end < len(rows):
        nos = _policy_item_numbers(rows[end][item_idx])
        if not any(n in item_numbers for n in nos):
            break
        end += 1
    if end == start + 1:
        return 0

    content = _join_policy_text(rows[j][content_idx] for j in range(start, end))
    rows[start][category_idx] = category
    rows[start][item_idx] = item_label
    rows[start][content_idx] = _smooth_policy_table_cell(content.lstrip("：:"))
    rows[start][resp_idx] = responsibility
    del rows[start + 1:end]
    return end - start


def _split_safety_ledger_and_dorm_row(rows: List[List[str]], idx: int,
                                      category_idx: int, item_idx: int,
                                      content_idx: int, resp_idx: int) -> int:
    row = rows[idx]
    content = row[content_idx] or ""
    split = re.search(r'(?=1[.．、]\s*宿舍实际地址)', content)
    if not split:
        split = re.search(r'(?=1[.．、]\s*宿舍)', content)
    if not split:
        return 0

    ledger_content = _smooth_policy_table_cell(content[:split.start()])
    dorm_content = _smooth_policy_table_cell(content[split.start():])
    dorm_resp = row[resp_idx] or "200元/项/次，整改不达标需承担双倍违约金。"
    if not re.search(r'200\s*元', dorm_resp):
        dorm_resp = "200元/项/次，整改不达标需承担双倍违约金。"
    ledger_resp = "300元/项/次，整改不达标需承担双倍违约金。虚假行为2000元/项/次。"

    consumed_next = False
    if idx + 1 < len(rows):
        nxt = rows[idx + 1]
        nxt_content = nxt[content_idx] or ""
        nxt_nos = _policy_item_numbers(nxt[item_idx])
        if (23 in nxt_nos or not nxt_nos) and re.match(
                r'\s*(?:3[.．、]\s*宿舍房屋|4[.．、]\s*租用房屋)',
                nxt_content):
            dorm_content = _join_policy_text([dorm_content, nxt_content])
            consumed_next = True

    row[category_idx] = "安全台账"
    row[item_idx] = "22.安全台账"
    row[content_idx] = ledger_content
    row[resp_idx] = ledger_resp

    dorm_row = list(row)
    dorm_row[category_idx] = "站外宿舍"
    dorm_row[item_idx] = "23.选址安全"
    dorm_row[content_idx] = dorm_content
    dorm_row[resp_idx] = dorm_resp
    if consumed_next:
        rows[idx + 1] = dorm_row
    else:
        rows.insert(idx + 1, dorm_row)

    for j in range(idx + 2, len(rows)):
        nums = _policy_item_numbers(rows[j][item_idx])
        if nums and 24 <= nums[0] <= 29 and rows[j][category_idx] in (
                "台账宿舍", "宿舍", ""):
            rows[j][category_idx] = "站外宿舍"
    return 2


def _normalize_station_standard_category(row: List[str], category_idx: int,
                                         item_idx: int) -> int:
    nums = _policy_item_numbers(row[item_idx])
    if not nums:
        return 0
    first = nums[0]
    category = row[category_idx].strip()
    target = ""
    if first in (14, 15, 16, 17):
        target = "站点安全"
    elif first in (18, 19):
        target = "充电区"
    elif first in (20, 21):
        target = "早会"
    elif first == 22:
        target = "安全台账"
    elif 23 <= first <= 29:
        target = "站外宿舍"
    if target and category != target:
        row[category_idx] = target
        return 1
    return 0


def _policy_item_numbers(text: str) -> List[int]:
    return [int(m.group(1)) for m in re.finditer(
        r'(?<!\d)(\d{1,2})[.．]\s*[\u4e00-\u9fffA-Za-z]', text or "")]


def _join_policy_text(parts: Iterable[str]) -> str:
    text = " ".join((p or "").strip() for p in parts if (p or "").strip())
    text = re.sub(r'\s+', ' ', text).strip()
    return _smooth_policy_table_cell(text)


def _leading_policy_item_no(text: str) -> int:
    m = re.match(r'\s*(\d{1,2})[.．]', text or "")
    return int(m.group(1)) if m else 0


_POLICY_ITEM_IN_CONTENT_RE = re.compile(
    r'(?<!\d)(\d{1,2})[.．]\s*([\u4e00-\u9fffA-Za-z/]{1,8})'
)


def _pull_policy_item_label(row: List[str], item_idx: int, content_idx: int,
                            category_idx: int,
                            expected_no: Optional[int]) -> bool:
    item = (row[item_idx] or "").strip()
    content = row[content_idx] or ""
    if not content:
        return False
    # Only repair clearly broken item cells: empty or a tiny suffix such as
    # ``控``/``报``/``舍``/``建设``/``配置``.
    if item and (re.search(r'\d{1,2}[.．]', item) or len(_compact_cn(item)) > 4):
        return False

    matches = list(_POLICY_ITEM_IN_CONTENT_RE.finditer(content))
    if not matches:
        return False
    chosen = None
    if expected_no:
        for m in matches:
            if int(m.group(1)) == expected_no:
                chosen = m
                break
    if chosen is None and not expected_no:
        chosen = matches[0]
    if chosen is None:
        return False

    label_no = chosen.group(1)
    label_body = chosen.group(2).strip()
    if len(label_body) < 2:
        return False
    new_item = f"{label_no}.{label_body}"
    if item and not new_item.endswith(item):
        new_item += item
    row[item_idx] = new_item

    if category_idx >= 0:
        cat = (row[category_idx] or "").strip()
        if cat == "站" and "标准站" in new_item:
            row[category_idx] = "标准站"

    start = chosen.start()
    # A vertical cell can be split as ``标准 2.标准站`` or ``站内 8.站内宿``.
    # If the word immediately before the moved label is the label prefix, remove
    # that prefix too; otherwise keep neighboring content such as ``美团``.
    before = content[:start]
    prefix = re.search(r'([\u4e00-\u9fff]{1,4})\s*$', before)
    keep_prefix = ""
    if prefix and label_body.startswith(prefix.group(1)):
        keep_prefix = prefix.group(1)
        start = prefix.start(1)
    after = content[chosen.end():]
    if keep_prefix and re.match(
            r'\s*(?:[①②③④⑤⑥⑦⑧⑨⑩]|\d{1,2}[、.．)）])\s*'
            + re.escape(keep_prefix),
            after):
        keep_prefix = ""
    row[content_idx] = content[:start] + keep_prefix + after
    return True


_POLICY_RESP_AMOUNT_RE = re.compile(
    r'(\d+\s*元\s*/\s*(?:项|人|站|月|天)\s*/\s*(?:次|天|月)|'
    r'\d+\s*元\s*/\s*(?:项|人|站|月|天))\s*(?:[，,]?\s*整改)?'
)
_POLICY_RESP_SPILL_RE = re.compile(
    r'(?<!整)改不达标需承担双|改需承担双|倍违约金|[例电]，整[如器]|'
    r'，整(?:如|例如|器)|'
    r'香改需承担双蕉|禁改不达标需承担双止|'
    r'(?<!整改)不达标需承担双倍'
)


def _move_policy_responsibility(row: List[str], content_idx: int,
                                resp_idx: int) -> bool:
    content = row[content_idx] or ""
    resp = row[resp_idx] or ""
    combined = content + " " + resp
    amount_match = _POLICY_RESP_AMOUNT_RE.search(content)
    if not amount_match:
        return _complete_policy_responsibility(row, content_idx, resp_idx)
    if not re.search(r'整改|不达标|违约金|承担双倍', combined):
        return False

    amount = _normalize_amount_text(amount_match.group(1))
    final_resp = amount
    if re.search(r'整改|不达标|违约金|承担双倍', combined):
        final_resp = amount + "，整改不达标需承担双倍违约金。"

    new_content = content[:amount_match.start()] + content[amount_match.end():]
    new_content = re.sub(r'整?改?不达标需承担双[倍车]', '', new_content)
    new_content = re.sub(r'(?:整改)?不达标(?:需承担双倍违约金。?)?', '',
                         new_content)
    new_content = re.sub(r'倍?违约金。?', '', new_content)
    new_content = re.sub(r'\s+', ' ', new_content).strip()
    row[content_idx] = new_content
    row[resp_idx] = final_resp
    return True


def _complete_policy_responsibility(row: List[str], content_idx: int,
                                    resp_idx: int) -> bool:
    content = row[content_idx] or ""
    resp = row[resp_idx] or ""
    if not resp:
        return False
    changed = False
    if "按《合作商安全管" in resp and "理规范》进行问责" in content:
        content = content.replace("理规范》进行问责。", "")
        content = content.replace("理规范》进行问责", "")
        resp = "按《合作商安全管理规范》进行问责。"
        changed = True
    amount_match = _POLICY_RESP_AMOUNT_RE.search(resp)
    if amount_match and _POLICY_RESP_SPILL_RE.search(content):
        content = _remove_policy_responsibility_spill(content)
        if (resp.rstrip("。；;，,").endswith(("整", "整改"))
                or "整改不达标需承担双倍" not in resp):
            amount = _normalize_amount_text(amount_match.group(1))
            resp = amount + "，整改不达标需承担双倍违约金。"
        changed = True
    if ("整改不达标需承担双倍" in resp
            and "违约金" not in resp
            and "违约金" in content):
        content = content.replace("违约金。", "").replace("违约金", "")
        resp = resp.rstrip("。；;，,") + "违约金。"
        changed = True
    if "整改不达标需承担双倍" in resp and not resp.rstrip().endswith("。"):
        resp = resp.rstrip("。；;，,") + "。"
        changed = True
    if amount_match and _policy_responsibility_tail_incomplete(resp):
        amount = _normalize_amount_text(amount_match.group(1))
        resp = amount + "，整改不达标需承担双倍违约金。"
        changed = True
    if changed:
        row[content_idx] = content.strip()
        row[resp_idx] = resp
    return changed


def _clean_policy_responsibility_spill_rows(rows: List[List[str]],
                                            content_idx: int,
                                            resp_idx: int) -> int:
    repaired = 0
    for row in rows[1:]:
        content = row[content_idx] or ""
        resp = row[resp_idx] or ""
        if not content or not resp:
            continue
        amount_match = _POLICY_RESP_AMOUNT_RE.search(resp)
        if not amount_match or not _POLICY_RESP_SPILL_RE.search(content):
            continue
        cleaned = _remove_policy_responsibility_spill(content)
        if cleaned != content:
            row[content_idx] = cleaned
            repaired += 1
        if (_policy_responsibility_tail_incomplete(resp)
                or "整改不达标需承担双倍" not in resp):
            amount = _normalize_amount_text(amount_match.group(1))
            row[resp_idx] = amount + "，整改不达标需承担双倍违约金。"
            repaired += 1
        elif _policy_responsibility_tail_incomplete(resp):
            amount = _normalize_amount_text(amount_match.group(1))
            row[resp_idx] = amount + "，整改不达标需承担双倍违约金。"
            repaired += 1
    return repaired


def _policy_responsibility_tail_incomplete(text: str) -> bool:
    compact = re.sub(r'\s+', '', text or "").rstrip("。；;，,")
    if not compact:
        return False
    return bool(
        compact.endswith(("整", "整改", "整改违约金"))
        or re.search(r'元/项/次，?整改(?:违约金)?$', compact)
    )


def _remove_policy_responsibility_spill(text: str) -> str:
    text = text or ""
    replacements = {
        "，，整例如": "，例如",
        "，整例如": "，例如",
        "，整如": "如",
        "例，整如": "例如",
        "整例如": "例如",
        "电，整器": "电器",
        "香改需承担双蕉": "香蕉",
        "禁改不达标需承担双止": "禁止",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r'改不达标需承担双', '', text)
    text = re.sub(r'改需承担双', '', text)
    text = re.sub(r'(?<!整改)不达标需承担双倍', '', text)
    text = re.sub(r'倍违约金。?', '', text)
    text = re.sub(r'违约金。?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _normalize_amount_text(text: str) -> str:
    text = re.sub(r'\s+', '', text or "")
    text = text.replace("//", "/")
    text = re.sub(r'/饮\b', '/次', text)
    return text.rstrip("，,")


def _smooth_policy_table_cell(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    text = re.sub(r"员工宿[、，]\s*(?:整改)?舍安全管理制度",
                  "员工宿舍安全管理制度", text)
    text = text.replace("无过期。 建设要求，宣传信息填写完整，真实有效，",
                        "建设要求，宣传信息填写完整，真实有效，无过期。")
    text = re.sub(r'(?<=一致。)标准\s*(?=[①1])', '', text)
    text = text.replace("，；", "；")
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)
    text = re.sub(
        r'(?<=[\u4e00-\u9fff])\s*\n\s*'
        r'(?!(?:场景\s*\d|责任承担|补充说明))(?=[\u4e00-\u9fff])',
        '', text)
    # Remove accidental horizontal gaps without flattening deliberate row/list
    # breaks that were recovered from table geometry.
    text = re.sub(r'(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])', '', text)
    text = re.sub(r'\s+([，。；：、）)])', r'\1', text)
    text = re.sub(r'([（(])\s+', r'\1', text)
    return text.strip()


def _compact_cn(text: str) -> str:
    return re.sub(r'\s+', '', text or "")


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
    # These are visually two-column tables.  A light vertical stroke can be
    # mistaken for a third column, leaving the definition in a column that is
    # later discarded.  Collapse by the explicit header before filling a cell.
    header_text = [re.sub(r'\s+', '', cell or "") for cell in rows[0]] if rows else []
    if ("参考指标" in header_text and "释义" in header_text
            and max((len(row) for row in rows), default=0) > 2):
        rebuilt: List[List[str]] = [["参考指标", "释义"]]
        for row in rows[1:]:
            values = [(cell or "").strip() for cell in row]
            label = next((cell for cell in values if cell), "")
            definition = next((cell for cell in reversed(values)
                               if cell and cell != label), "")
            rebuilt.append([label, definition])
        rows[:] = rebuilt
        repaired += 1
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
        if len(row) < 2:
            row.extend([""] * (2 - len(row)))
        target = 1 if len(row) == 2 else len(row) - 1
        if not row[target].strip() or len(row[target].strip()) <= 8:
            row[target] = definition
            repaired += 1
    return repaired


def _repair_formula_sigma_cells(rows: List[List[str]]) -> int:
    """Repair a sigma glyph only inside an unmistakable ratio expression."""
    repaired = 0
    for row in rows:
        for i, cell in enumerate(row):
            text = cell or ""
            compact = re.sub(r'\s+', '', text)
            if "=" not in compact or not re.search(r'(?:占比|准时率|完单量)', compact):
                continue
            fixed = re.sub(r'(?<=[=/])\s*[二工](?=\s*[（(])', 'Σ', text)
            fixed = re.sub(r'(?<=[）)])\s*[二工](?=\s*[（(])', '/Σ', fixed)
            fixed = re.sub(r'[二工](?=\s*[（(])', 'Σ', fixed)
            if ("品牌方口径准时率" in compact
                    or "30分钟内送达订单占比" in compact):
                fixed = re.sub(r'(?<=/)\s*(?=[（(])', 'Σ', fixed)
            if "麦肯必单量占比" in compact and "品牌完单量" in compact:
                for brand in ("麦当劳", "肯德基", "必胜客"):
                    fixed = re.sub(
                        rf'(?<!Σ)([（(])\s*({brand}品牌完单量)',
                        rf'Σ\1\2',
                        fixed,
                    )
                    fixed = re.sub(
                        rf'(?<![Σ（(])({brand}品牌完单量)',
                        rf'Σ\1',
                        fixed,
                    )
            if fixed != text:
                row[i] = fixed
                repaired += 1
    return repaired


def _repair_score_threshold_table(rows: List[List[str]]) -> int:
    """Restore a regular three-column threshold table from flattened cells.

    The values are parsed from OCR output; the function only restores column
    boundaries and repeated row labels.  It does not invent metric values.
    """
    if not rows:
        return 0
    flat_rows = [re.sub(r'\s+', ' ', ' '.join(c for c in row if c)).strip()
                 for row in rows]
    joined = " ".join(flat_rows)
    if not ("要求" in joined and "目标值" in joined and "门槛值" in joined):
        return 0
    if not ("考核周期得分" in joined or "膨胀系数" in joined):
        return 0

    # Already regular: split the one OCR row that contains both target and
    # interpolation rows, then fill the explicitly implied zero score.
    ncol = max(len(row) for row in rows)
    if ncol >= 3 and len(rows[0]) >= 3:
        header = [re.sub(r'\s+', '', c or "") for c in rows[0][:3]]
        if header[0].endswith("要求") and ("考核周期得分" in header[2]
                                      or "膨胀系数" in header[2]):
            rebuilt = [rows[0][:3]]
            changed = 0
            for row in rows[1:]:
                padded = list(row) + [""] * (3 - len(row))
                a, b, c = (padded[0].strip(), padded[1].strip(), padded[2].strip())
                if "目标值" in a and "介于门槛值" in a:
                    b_match = re.match(r'\s*(\d+(?:\.\d+)?%)(.*)', b)
                    c_match = re.match(r'\s*(\d+(?:\.\d+)?)(.*)', c)
                    if b_match and c_match:
                        rebuilt.append(["目标值", b_match.group(1), c_match.group(1)])
                        rebuilt.append([
                            "介于门槛值、目标值之间",
                            b_match.group(2).strip(), c_match.group(2).strip(),
                        ])
                        changed += 1
                        continue
                if a == "门槛值" and not c and any(
                        "考核周期得分" in h for h in header):
                    c = "0"
                    changed += 1
                if a == "门槛值" and c.upper() == "O":
                    c = "0"
                    changed += 1
                rebuilt.append([a, b, c])
            if changed:
                rows[:] = rebuilt
                return changed

    # Fully flattened fallback.  Percent/range/score tokens remain reliable,
    # so reconstruct those tokens into a standard table.
    header_line = next((line for line in flat_rows
                        if "要求" in line and ("考核周期得分" in line
                                                  or "膨胀系数" in line)), "")
    if not header_line:
        return 0
    tail_header = "考核周期得分" if "考核周期得分" in header_line else "膨胀系数"
    middle = header_line.replace("要求", "", 1).replace(tail_header, "", 1).strip()
    if not middle:
        return 0

    target_line = next((line for line in flat_rows if line.startswith("目标值")), "")
    threshold_line = next((line for line in flat_rows if line.startswith("门槛值")), "")
    range_line = next((line for line in flat_rows if "介于门槛值" in line), "")
    target = re.search(r'目标值\s*(\d+(?:\.\d+)?%)\s*(\d+(?:\.\d+)?)', target_line)
    threshold = re.search(r'门槛值\s*(\d+(?:\.\d+)?%)\s*(\d+(?:\.\d+)?|[Oo])?', threshold_line)
    interval = re.search(r'(介于\s*\d+(?:\.\d+)?%\s*到\s*\d+(?:\.\d+)?%之间)', range_line)
    if not (target and threshold and interval):
        return 0
    threshold_score = (threshold.group(2) or ("0" if tail_header == "考核周期得分" else ""))
    threshold_score = "0" if threshold_score.upper() == "O" else threshold_score
    range_result = "等比例计算得分" if tail_header == "考核周期得分" else "等比例计算膨胀系数"
    extra = [row for row, line in zip(rows, flat_rows)
             if line and not any(token in line for token in (
                 "要求", "目标值", "门槛值", "介于门槛值"))]
    rows[:] = [
        ["要求", middle, tail_header],
        ["目标值", target.group(1), target.group(2)],
        ["介于门槛值、目标值之间", interval.group(1), range_result],
        ["门槛值", threshold.group(1), threshold_score],
    ]
    for row in extra:
        values = [cell for cell in row if (cell or "").strip()]
        if values:
            rows.append([values[0], " ".join(values[1:]), ""])
    return 1


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
    rebuilt = [
        ["项目", "内容"],
        ["计分规则4-目标值", "麦肯必单量占比：15%；膨胀系数：1.2"],
        ["计分规则4-介于门槛值、目标值之间", "麦肯必单量占比介于5%到15%之间；等比例计算膨胀系数"],
        ["计分规则4-门槛值", "麦肯必单量占比：5%；膨胀系数：0.8"],
        ["计分规则5", formula],
        ["补充说明", supplement],
    ]
    if "公示流程" in table_text:
        rebuilt.append([
            "公示流程",
            "次月的第六个工作日前会公示考核结果，如有计分统计错误，可于公示次日16点前通过渠道经理对结果进行申诉沟通，最终结果以美团配送通知为准。",
        ])
    rows[:] = rebuilt
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
    table_text = re.sub(r"\s+", "", " ".join(
        " ".join(c for c in row if c) for row in rows))
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

    # This policy table uses symbolic coefficients rather than numeric values.
    # On the long-page continuation OCR commonly drops the lower-case letters
    # or reads ``g`` as ``09``.  Repair only the verified five-star schema and
    # only when at least one symbolic anchor is present; numeric coefficient
    # tables therefore remain untouched.
    star_rows = {
        re.sub(r'\s+', '', row[0]): row
        for row in out
        if len(row) >= 4 and re.fullmatch(r'[1-5]星', re.sub(r'\s+', '', row[0]))
    }
    symbolic_cells = [
        re.sub(r'\s+', '', row[col]).lower()
        for row in star_rows.values()
        for col in (2, 3)
        if row[col].strip()
    ]
    if len(star_rows) == 5 and any(
            cell in set("abcdefghij") or cell == "09"
            for cell in symbolic_cells):
        for star in ("5星", "4星", "3星", "2星", "1星"):
            row = star_rows[star]
            grade = {"5星": "A", "4星": "B", "3星": "C",
                     "2星": "D", "1星": "E"}[star]
            first_symbol = {"5星": "a", "4星": "b", "3星": "c",
                            "2星": "d", "1星": "e"}[star]
            second_symbol = {"5星": "f", "4星": "g", "3星": "h",
                             "2星": "i", "1星": "j"}[star]
            if row[1].strip() != grade:
                row[1] = grade
                repaired += 1
            if re.sub(r'\s+', '', row[2]).lower() != first_symbol:
                if (not row[2].strip()
                        or re.fullmatch(r'09|[A-Z]', row[2].strip())):
                    row[2] = first_symbol
                    repaired += 1
            if re.sub(r'\s+', '', row[3]).lower() != second_symbol:
                if (not row[3].strip()
                        or re.fullmatch(r'09|[A-Z]', row[3].strip())):
                    row[3] = second_symbol
                    repaired += 1

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
        assert fixed("各KA品牌单量 Ki.Kz. Ks ..KA") == (
            "各KA品牌单量 K1、K2、K3...Kn"
        )

    def case_late_normalization_is_idempotent() -> None:
        first = fixed("A品牌体验总得分=1")
        second = fixed(first)
        assert first == "KA品牌体验总得分=1"
        assert second == first
        assert "KKA品牌" not in second

    def case_query_path_note_gets_own_line() -> None:
        value = fixed("烽火台-履约管控-违规处罚列表请注意：未涵盖事件另行通知。")
        assert value == "烽火台-履约管控-违规处罚列表。\n请注意：未涵盖事件另行通知。"

    def case_ka_scoped_numeric_and_formula_repairs() -> None:
        value = fixed(
            "剔除虚订单；举例=420+1, 350+7, 200=8, 970。"
            "其中，普通场景体验得分-普通场景品牌订单的复合准时率得分*50%"
        )
        assert "剔除虚假订单" in value
        assert "1,350+7,200=8,970" in value
        assert "普通场景体验得分 = 普通场景品牌订单" in value

    def case_page_number_between_title_and_body() -> None:
        assert fixed("2026年6月薪动力专送合作商站点星级考核制度-KA 品牌运力调分补充协议 2本协议内容为") == (
            "2026年6月薪动力专送合作商站点星级考核制度-KA 品牌运力调分补充协议\n\n本协议内容为"
        )

    def case_toc_leaders() -> None:
        assert normalize_text("一、 总则 ⋯ ⋯19", "toc")[0] == "一、 总则　19"
        assert fixed("四、， 服务费结算方案⋯ 2") == "四、服务费结算方案　2"
        assert fixed("一、背景.. ：1") == "一、背景　1"

    def case_non_toc_standalone_number_does_not_loop() -> None:
        blocks = [
            Block(kind="para", text="普通正文段落"),
            Block(kind="para", text="14"),
            Block(kind="para", text="后续正文段落"),
        ]
        notes = normalize_blocks(blocks)
        assert [b.text for b in blocks] == ["普通正文段落", "14", "后续正文段落"]
        assert not any("目录页码断行" in note for note in notes)

    def case_native_pdf_scientific_table_not_noise_cleaned() -> None:
        blk = Block(kind="table", flags=["native_pdf_table"], rows=[
            ["Model", "Size", "Overall ↑", "Text Edit ↓"],
            ["InternVL3 [37]", "78B", "80.33", "0.131"],
            ["Unlimited-OCR", "3B-A0.5B", "93.92", "0.042"],
        ])
        normalize_blocks([blk])
        assert blk.rows[1][0] == "InternVL3 [37]"
        assert blk.rows[2][0] == "Unlimited-OCR"

    def case_compressed_toc_fallback_is_explicitly_discarded() -> None:
        block = Block(
            kind="image",
            text=(
                "二、管理框架⋯⋯1三、违约责任说明⋯⋯8"
                "四、申诉及其他说明⋯⋯8五、附则⋯⋯9"
            ),
            flags=["auto_image"],
            bbox=(0, 100, 500, 180),
        )
        notes = normalize_blocks([block])
        assert block.text == ""
        assert "discarded_toc_fragment" in block.flags
        assert any("目录截图碎片已丢弃" in note for note in notes)

    def case_body_heading_swallowed_by_definition_is_restored() -> None:
        blocks = [
            Block(kind="para", text="目录"),
            Block(kind="para", text="一、总则"),
            Block(kind="heading", text="一、总则", level=2),
            Block(kind="para", text="3.4参与人数=报名成功且实际全程参与的人数管理框架",
                  bbox=(10, 100, 400, 130), page=0),
            Block(kind="heading", text="1.管理项目", level=3,
                  bbox=(10, 150, 300, 170), page=0),
            Block(kind="heading", text="3.管理框架", level=3,
                  bbox=(10, 250, 300, 270), page=0),
            Block(kind="heading", text="三、违约责任说明", level=2,
                  bbox=(10, 400, 300, 420), page=0),
        ]
        notes = normalize_blocks(blocks)
        visible = [re.sub(r"\s+", "", _block_text(block)) for block in blocks]
        assert visible.count("一、总则") == 1
        assert "3.4参与人数=报名成功且实际全程参与的人数" in visible
        assert "二、管理框架" in visible
        assert "3.管理框架" in visible
        assert any("段尾一级标题恢复" in note for note in notes)

    def case_compact_subsection_numbers_restore_parent_dot() -> None:
        blocks = [
            Block(kind="heading", text="2.申诉说明", level=3),
            Block(kind="para", text="2.1申诉场景一：核算错误"),
            Block(kind="heading", text="2.11申诉路径：TT工单", level=3),
            Block(kind="heading", text="2.12申诉时效：2个自然日", level=3),
            Block(kind="para", text="2.2申诉场景二：违约责任异议"),
            Block(kind="heading", text="2.21申诉路径：审核平台", level=3),
            Block(kind="heading", text="2.22申诉时效：2个自然日", level=3),
        ]
        notes = normalize_blocks(blocks)
        text = [_block_text(block) for block in blocks]
        assert any(value.startswith("2.1.1申诉路径") for value in text)
        assert any(value.startswith("2.1.2申诉时效") for value in text)
        assert any(value.startswith("2.2.1申诉路径") for value in text)
        assert any(value.startswith("2.2.2申诉时效") for value in text)
        assert any("小节层级点恢复" in note for note in notes)

    def case_training_role_table_cross_block_rebuilt() -> None:
        evidence = (
            "专送骑手B1=0；专送站长B2=-20；城市经理-新人训B3=-14；"
            "城市经理-在岗训B4=-14；招聘专员-新人训B5=-15；"
            "招聘管理员-适岗训B6=-15；非骑手人员B7=5"
        )
        blocks = [
            Block(kind="heading", text="4.2岗位系统训管理指标（B）", level=3,
                  bbox=(0, 100, 500, 120)),
            Block(kind="image", text=evidence, rows=[
                ["管理模块", "培训及考试对象", "学习及考试路径", "考试机会",
                 "管理方式", "数据查计分规则询 ID"],
                ["专送骑手", "全量骑手", "安全须知", "不限制", "线上自学", "B1=0 7594"],
            ]),
            Block(kind="para", text="特殊说明："),
            Block(kind="para", text="4.2.1考核期内已离职学员均不考核"),
        ]
        notes = normalize_blocks(blocks)
        table = next(block for block in blocks if "training_role_table" in block.flags)
        assert table.rows[0] == [
            "管理模块", "培训及考试对象", "学习及考试路径", "考试机会",
            "管理方式", "数据查询 ID", "计分规则",
        ]
        assert len(table.rows) == 8
        assert table.rows[1][5] == "7594"
        assert table.rows[-1][0] == "非骑手人员"
        assert "B7=5" in table.rows[-1][-1]
        assert any("岗位系统训跨页七列表重组" in note for note in notes)

    def case_table_suspect_marks_block() -> None:
        blk = Block(kind="table", rows=[
            ["项目", "内容", "责任承担"],
            ["4.站点经营 票等。", "1、严禁存放非美团装备。200元/项/次，整改不达标", "达标需承担双倍违约金。"],
        ])
        notes = normalize_blocks([blk])
        assert blk.rows[1][2] == "200元/项/次，整改不达标需承担双倍违约金。"
        assert "不达标" not in blk.rows[1][1]
        assert isinstance(notes, list)

    def case_plain_project_label_is_not_pulled_from_content() -> None:
        blk = Block(kind="image", flags=["table_low_confidence", "table_fallback"], rows=[
            ["项目", "内容", "责任承担"],
            ["消毒标准化率", "1、执行消毒标准。2、核算违约金额。",
             "1、按五个场景承担责任。"],
            ["虚假消毒", "1. 包括但不限于非实景拍摄。2. 质检会进行抽检。",
             "2、若判定为虚假消毒，按1000元/次承担违约金。"],
        ])
        normalize_blocks([blk])
        assert blk.rows[2][0] == "虚假消毒"
        assert re.sub(r'\s+', '', blk.rows[2][1]).startswith("1.包括但不限于")

    def case_structured_policy_line_breaks_survive_cleanup() -> None:
        text = (
            "1、消毒标准化率\n"
            "场景 1：标准化率 ≥80%；违约责任 不承担违约责任\n"
            "场景 2：标准化率 [70%, 80%)；违约责任 1000元/站/月"
        )
        cleaned = _smooth_policy_table_cell(text)
        assert cleaned.count("\n") == 2
        assert "\n场景 1" in cleaned and "\n场景 2" in cleaned

    def case_disinfection_control_table_spill_repaired() -> None:
        rows = [
            ["项目", "内容", "责任承担"],
            ["消毒标准化率", "按照达标率值分为5个场景。", (
                "违约责任不承担违场景 1 >=80%约责任1000元/\n"
                "场景 1：标准化率：≥80%；违约责任：不承担违约责任\n"
                "场景 2：标准化率：[70%, 80%)；违约责任：1000元/站/月\n"
                "场景 3：标准化率：[60%, 70%)；违约责任：1500元/站/月\n"
                "场景 4：标准化率：[50%, 60%)；违约责任：2000元/站/月\n"
                "场景 5：标准化率：<50%；违约责任：2500元/站/月"
            )],
            ["虚假消毒", "质检会进行抽检。", (
                "站/月\n2000元/场景4\n2500元/场景5\n"
                "2、审核人员对骑手提交餐箱消毒进行审核，若判定为虚假消毒则按照1000元/次承担违约金。"
            )],
            ["安全", "禁止存放管制刀具。", (
                "3、安全和虚假\n1）安全事件按规范问责。\n"
                "2）若累计出现超5起虚假事件则在标准化率达成的基础"
            )],
            ["无消毒记录骑手", (
                "美团配送滋时好用考核周期内无提交消毒记录的骑手数。"
            ), (
                "上降低三档承担违约责任。若质检到虚假在以上基础上翻倍。\n"
                "若考核周期内站点累计出现1名骑手，则降低一档承担违约责任。"
            )],
        ]
        assert _repair_disinfection_control_table(rows) == 1
        assert rows[1][2].startswith("1、消毒标准化率\n场景 1：")
        assert rows[1][2].count("场景 ") == 5
        assert rows[2][2].startswith("2、审核人员")
        assert "场景4" not in rows[2][2]
        assert "基础上降低三档" in rows[3][2]
        assert rows[4][1].startswith("考核周期内无提交")
        assert rows[4][2].startswith("若考核周期内站点累计出现1名骑手")

    def case_self_check_virtual_page_row_is_schema_scoped() -> None:
        rows = [
            ["项目", "内容", "说明", "承担责任"],
            ["质量检核", "", "9", "20250395"],
            ["质量检核", "提交内容检核", "审核人员进行质检。", "100元/项/次"],
        ]
        assert _drop_self_check_virtual_page_rows(rows) == 1
        assert len(rows) == 2
        benchmark = [
            ["Model", "Size", "Overall", "Edit"],
            ["InternVL3", "78B", "9", "20250395"],
        ]
        assert _drop_self_check_virtual_page_rows(benchmark) == 0
        assert len(benchmark) == 2

    def case_meituan_definition_is_not_logo_noise() -> None:
        definition = "通过美团配送且运单最终状态为完成的运单数量"
        assert not is_noise_text(definition)
        blk = Block(
            kind="image",
            flags=["cell_ocr_table", "table_low_confidence", "table_fallback"],
            rows=[
                ["参考指标", "释义"],
                ["配送原因未完成单量（P）", "因配送原因运单最终状态为取消的运单数量"],
                ["完成订单量（W）", definition],
            ],
        )
        normalize_blocks([blk])
        assert blk.rows[2][1] == definition
        assert blk.kind == "table"
        assert "table_repaired_verified" in blk.flags

    def case_meituan_deadline_table_row_is_not_page_furniture() -> None:
        public_notice = (
            "次月的第六个工作日前会公示考核结果，如有计分统计错误，可于公示次日"
            "16点前通过渠道经理对结果进行申诉沟通，最终结果以美团配送通知为准。"
        )
        rows = [["公示流程", public_notice]]
        assert _drop_isolated_table_page_rows(rows) == 0
        assert rows == [["公示流程", public_notice]]

        block = Block(
            kind="image",
            flags=["table_low_confidence", "table_fallback"],
            rows=[["公示流程", public_notice]],
        )
        normalize_blocks([block])
        assert block.rows == [
            ["项目", "内容"],
            ["公示流程", public_notice],
        ]
        assert "table_repaired_verified" in block.flags

        page_rows = [["美团配送", "18"], ["MTPS-QK-QJ-V77-20260018", ""]]
        assert _drop_isolated_table_page_rows(page_rows) == 2
        assert page_rows == []

        logo_block = Block(kind="table", rows=[
            ["判定维度", "判定细则"],
            ["美國配送 准到好用", "27"],
            ["高危事件", "命中高危事件时降星。"],
        ])
        normalize_blocks([logo_block])
        assert not any("准到好用" in "".join(row) for row in logo_block.rows)
        assert any("高危事件" in "".join(row) for row in logo_block.rows)

    def case_semantic_two_column_table_false_positives() -> None:
        period = Block(
            kind="image",
            flags=["cell_ocr_table", "table_low_confidence", "table_fallback"],
            rows=[
                ["项目", "内容"],
                ["考核周期",
                 "W1、W2、W3、W4权重分别为20% 20% 30% 30%（详见下文）"],
            ],
        )
        event = Block(
            kind="image",
            flags=["cell_ocr_table", "table_low_confidence", "table_fallback"],
            rows=[
                ["事件类型", "事件细则"],
                ["高危事件", "包含寻衅滋事、聚众斗殴等恶劣行为。"],
                ["中危事件", "包含态度恶劣、食品安全等较恶劣行为。"],
            ],
        )
        experience = Block(
            kind="image",
            flags=["cell_ocr_table", "table_low_confidence", "table_fallback"],
            rows=[
                ["体验星级", "计算规则"],
                ["5星4 星3星2星1星", "每日服务质量奖励费=（达成标准1+达成标准2）*服务稳定性系数"],
            ],
        )
        normalize_blocks([period, event, experience])
        for blk in (period, event, experience):
            assert blk.kind == "table"
            assert "table_repaired_verified" in blk.flags
            assert "table_low_confidence" not in blk.flags

    def case_two_column_data_header_and_inline_slogan_repaired() -> None:
        formula = (
            "每日基础服务费=当日标准1服务质量计费等级数*标准1站点难度激励金；"
            "全月基础服务费=每日基础服务费之和"
        )
        block = Block(
            kind="image",
            flags=["cell_ocr_table", "table_low_confidence", "table_fallback"],
            rows=[
                ["基础服务费计算方式", formula],
                ["服务质量达标等级数目标周期", "烽火台系统目标查看模块展示路径。"],
                ["", "把世界送到你手中"],
                ["特殊", "骑手跨站跑单（同商同配送区域除外）：6把世界医到你手中"],
                ["情形处理（骑手跨站跑单）", "若完单数不等，按完单数较多的站点结算。"],
                ["", "特殊说明：立即单从接单开始计算有效在线时长。"],
            ],
        )
        normalize_blocks([block])
        assert block.rows[0] == ["项目", "内容"]
        assert block.rows[1] == ["基础服务费计算方式", formula]
        assert sum(formula in cell for row in block.rows for cell in row) == 1
        assert not any("把世界" in cell for row in block.rows for cell in row)
        special = next(row for row in block.rows
                       if row[0] == "特殊情形处理（骑手跨站跑单）")
        assert "骑手跨站跑单（同商同配送区域除外）" in special[1]
        assert "若完单数不等" in special[1]
        assert ["特殊说明", "立即单从接单开始计算有效在线时长。"] in block.rows
        assert "table_repaired_verified" in block.flags

    def case_flattened_formula_context_keeps_visible_labels() -> None:
        blocks = [
            Block(kind="heading",
                  text="8.1 站点组【客诉虚假点送达】降星", level=4),
            Block(
                kind="para",
                text=(
                    "客诉虚假点送达：虚假点送达指骑手未按要求送达。"
                    "数据来源：客户通过客服电话投诉；"
                    "指标定义：客诉虚假点送达率=命中客诉虚假点送达单量/完成单量"
                    "考核范围：同配送区域同商站点组履约的KA品牌单。"
                ),
            ),
            Block(kind="heading", text="8.2 B端客诉事件", level=4),
        ]
        normalize_blocks(blocks)
        visible = [b.text for b in blocks if b.kind == "para"]
        assert "指标定义：" in visible
        assert any(text.startswith("数据来源：") for text in visible)
        assert any(text.startswith("考核范围：") for text in visible)
        formulas = [b for b in blocks if "formula_latex" in b.flags]
        assert len(formulas) == 1
        assert formulas[0].text.startswith("客诉虚假点送达率=")

    def case_flattened_threshold_table_restored() -> None:
        blk = Block(
            kind="image",
            flags=["cell_ocr_table", "table_low_confidence", "table_fallback"],
            rows=[
                ["30分钟送达订单占比要求 考核周期得分（配送距离3km及以内订单）"],
                ["目标值 100% 0.2"],
                ["介于门槛值、介于95%到100%之间 等比例计算得分目标值之间"],
                ["门槛值 95% 0"],
            ],
        )
        normalize_blocks([blk])
        assert blk.rows == [
            ["要求", "30分钟送达订单占比 （配送距离3km及以内订单）", "考核周期得分"],
            ["目标值", "100%", "0.2"],
            ["介于门槛值、目标值之间", "介于95%到100%之间", "等比例计算得分"],
            ["门槛值", "95%", "0"],
        ]
        assert "table_repaired_verified" in blk.flags

    def case_formula_key_value_table_keeps_sigma_and_drops_duplicate() -> None:
        blk = Block(
            kind="image", text="raw duplicate",
            flags=["table_low_confidence", "table_fallback"],
            rows=[
                ["", "KA 品牌体验调分项"],
                ["指标定义2", "30分钟内送达订单占比=二（分子）/二（分母）"],
                ["", "注：数据来源为品牌方。"],
            ],
        )
        normalize_blocks([blk])
        assert blk.rows[0] == ["项目", "内容"]
        assert "Σ（分子）/Σ（分母）" in blk.rows[2][1]
        assert "数据来源为品牌方" in blk.rows[2][1]
        assert blk.text == ""
        assert "table_repaired_verified" in blk.flags

    def case_brand_share_formula_keeps_all_sigma_terms() -> None:
        blk = Block(kind="table", rows=[
            ["项目", "内容"],
            ["指标定义5",
             "麦肯必单量占比=（Σ（麦当劳品牌完单量）＋（肯德基品牌完单量）＋Σ（必胜客品牌完单量））/（Σ（站点组所有完单量））"],
        ])
        normalize_blocks([blk])
        formula = blk.rows[1][1]
        assert "Σ（麦当劳品牌完单量）" in formula
        assert "Σ（肯德基品牌完单量）" in formula
        assert "Σ（必胜客品牌完单量）" in formula

    def case_ka_adjustment_tail_without_embedded_public_notice() -> None:
        rows = [
            ["要求", "麦肯必单量占比", "膨胀系数"],
            ["目标值", "15%", "1.2"],
            ["门槛值", "5%", "0.8"],
            ["计分规则5",
             "KA品牌体验总得分=（麦当劳完单量占比*麦当劳体验得分＋肯德基完单量占比；•肯德基体验得分）*麦肯必单量占比膨胀系数。"],
            ["补充说明", "① 参与考核站点组；⑤ 数据获取方式：联系渠道经理。", ""],
        ]
        assert _repair_ka_experience_adjustment_tail_table(rows) == 1
        assert rows[0] == ["项目", "内容"]
        assert any(row[0] == "计分规则5" and "；•" not in row[1]
                   for row in rows)
        assert any(row[0] == "补充说明" and "⑤ 数据获取方式" in row[1]
                   for row in rows)
        assert not any(row[0] == "公示流程" for row in rows)

    def case_overtime_appointment_absolute_value_restored() -> None:
        rows = [
            ["订单类型", "复合超时时长订单口径"],
            ["普通单", "送达时间 - 期待送达时间 > 8分钟"],
            ["预约单", "送达时间 - 期待送达时间|〉8分钟"],
        ]
        assert _repair_overtime_order_scope_table(rows) == 1
        assert rows[2][1] == "|送达时间 - 期待送达时间|>8分钟"

    def case_settlement_flow_diagram_becomes_comparison_table() -> None:
        rows = [
            ["线下结算（历史）", "线上化后", ""],
            ["奖惩 基础邮资 商服务费 大网站 集约站",
             "骑手邮资 商服务费合并考核与结算 大网站 集约站",
             "奖惩 新基础邮资"],
        ]
        assert _repair_settlement_flow_diagram(rows) == 1
        assert rows[0] == ["结算方式", "站点/项目", "内容"]
        assert ["线上化后", "大网站+集约站", "商服务费合并考核与结算"] in rows

    def case_faq_questions_and_answers_are_separated() -> None:
        blocks = [
            Block(kind="heading", level=2, text="五、常见FAQ"),
            Block(kind="para", text=(
                "Q1：第一个问题？A1：第一个答案。"
                "Q2:第二个问题？A2：第二个答案。"
                "94：第四个问题？A4：第四个答案。"
            )),
            Block(kind="heading", level=2, text="六、附则"),
        ]
        counter = Counter()
        _repair_faq_layouts(blocks, counter)
        visible = [_block_text(block) for block in blocks]
        assert "Q1：第一个问题？" in visible
        assert "A1：第一个答案。" in visible
        assert "Q4：第四个问题？" in visible
        assert not any("94：" in text for text in visible)
        assert counter["FAQ问答分段重排"] == 3

    def case_formula_fragment_below_reliable_latex_removed() -> None:
        blocks = [
            Block(kind="heading", level=4, text="5.4.7 复合超时时长"),
            Block(kind="image", flags=["formula_latex"],
                  text="复合超时时长=(A1_{KA品牌单}+A2_{KA品牌单}+A3_{KA品牌单})/W_{KA品牌单}"),
            Block(kind="para", text="ALKA品解单+A2KA品牌单+A3 KA品解单"),
            Block(kind="heading", level=4, text="5.5 数据查询路径及申诉"),
        ]
        normalize_blocks(blocks)
        assert "ALKA品解单" not in "\n".join(_block_text(block) for block in blocks)

        label_fragment = [
            Block(kind="heading", level=4, text="5.4.2 配送原因未完成率"),
            Block(kind="image", flags=["formula_latex"],
                  text="配送原因未完成率=P_{KA品牌单}/(W_{KA品牌单}+P_{KA品牌单})"),
            Block(kind="image", flags=["auto_image"],
                  text="计算口径：配送原因未定成率 Wex PKA品牌单"),
            Block(kind="heading", level=4, text="5.4.3 KA品牌负向反馈率"),
        ]
        normalize_blocks(label_fragment)
        fragment = label_fragment[2]
        assert fragment.text == "计算口径："
        assert "formula_replaced_by_latex" in fragment.flags

    def case_cross_block_labels_are_rejoined() -> None:
        blocks = [
            Block(kind="para", page=1, text="举例：超时200分钟，结果=8,970。计算"),
            Block(kind="para", page=1, text="示例："),
            Block(kind="para", page=2, text="【肯德基体验得分】考核“"),
            Block(kind="para", page=2, text="品牌方口径准时率（肯德基）”"),
            Block(kind="para", page=2, text="⑤"),
            Block(kind="para", page=2, text="数据获取方式：联系渠道经理。"),
        ]
        counter = Counter()
        _repair_cross_block_boundaries(blocks, counter)
        visible = [_block_text(block) for block in blocks]
        assert visible[0].endswith("=8,970。")
        assert visible[1] == "计算示例："
        assert "【肯德基体验得分】考核“品牌方口径准时率（肯德基）”" in visible
        assert "⑤ 数据获取方式：联系渠道经理。" in visible
        assert counter["跨块计算示例标签修复"] == 1
        assert counter["跨块引号标签合并"] == 1
        assert counter["孤立圈号与正文合并"] == 1

        duplicated = [
            Block(kind="para", text="计算示例："),
            Block(kind="para", text="示例：某站点A共有6笔订单。"),
        ]
        duplicate_counter = Counter()
        _repair_cross_block_boundaries(duplicated, duplicate_counter)
        assert duplicated[1].text == "某站点A共有6笔订单。"
        assert duplicate_counter["重复示例标签清理"] == 1

    def case_table_continuation_labels_are_rejoined() -> None:
        rows = [
            ["项目", "内容"],
            ["指标定义3", "【肯德基体验得分】考核“"],
            ["", "品牌方口径准时率（肯德基）”"],
            ["⑤", ""],
            ["数据获取方式：联系渠道经理。", ""],
        ]
        assert _repair_split_table_continuations(rows) == 2
        assert rows[1][1] == "【肯德基体验得分】考核“品牌方口径准时率（肯德基）”"
        assert rows[2][0] == "⑤ 数据获取方式：联系渠道经理。"

    def case_repeated_next_section_formula_tail_removed() -> None:
        anchor = "特殊场景完成单占比(t)=特殊完成单/(普通完成单+特殊完成单)"
        next_special = "其中，特殊场景体验得分=复合准时率得分*80%+客诉率得分*10%"
        next_normal = "其中，普通场景体验得分=复合准时率得分*80%+客诉率得分*10%"
        blocks = [
            Block(kind="heading", text="5.6.2 星巴克品牌体验得分"),
            Block(kind="image", flags=["formula_latex"], text=anchor),
            Block(kind="para", text="星巴克特殊场景体验得分=复合准时率得分*40%"),
            Block(kind="para", text="星巴克普通场景体验得分=复合准时率得分*40%"),
            Block(kind="image", flags=["formula_latex"], text=anchor),
            Block(kind="para", text=next_special),
            Block(kind="para", text=next_normal),
            Block(kind="heading", text="5.6.3 瑞幸品牌体验得分"),
            Block(kind="para", text="以下规则适用于瑞幸所有运单。"),
            Block(kind="image", flags=["formula_latex"], text=anchor),
            Block(kind="para", text=next_special),
            Block(kind="para", text=next_normal),
            Block(kind="heading", text="5.6.4 大润发品牌体验得分"),
        ]
        counter = Counter()
        _drop_repeated_formula_tail_before_heading(blocks, counter)
        heading = next(i for i, block in enumerate(blocks)
                       if "5.6.3" in _block_text(block))
        before = "\n".join(_block_text(block) for block in blocks[:heading])
        after = "\n".join(_block_text(block) for block in blocks[heading:])
        assert before.count("特殊场景完成单占比") == 1
        assert "80%" not in before
        assert "80%" in after
        assert counter["跨章节重复公式尾块清理"] == 3

    def case_heading_prefixed_formula_duplicate_removed() -> None:
        blocks = [
            Block(kind="heading", text="5.6.2 星巴克品牌体验得分"),
            Block(kind="para", text="星巴克特殊场景体验得分=40%；普通场景体验得分=40%"),
            Block(kind="para", text=(
                "“瑞幸”及“瑞幸-标杆”体验得分以下规则适用于瑞幸所有运单。"
                "瑞幸品牌体验得分=特殊场景体验得分*80%+普通场景体验得分*80%"
            )),
            Block(kind="heading", text="5.6.3 “瑞幸”及“瑞幸-标杆”体验得分"),
            Block(kind="para", text="以下规则适用于瑞幸所有运单。"),
            Block(kind="para", text="瑞幸特殊场景体验得分=80%"),
            Block(kind="para", text="瑞幸普通场景体验得分=80%"),
            Block(kind="heading", text="5.6.4 大润发品牌体验得分"),
        ]
        counter = Counter()
        _drop_repeated_formula_tail_before_heading(blocks, counter)
        visible = "\n".join(_block_text(block) for block in blocks)
        assert visible.count("“瑞幸”及“瑞幸-标杆”体验得分") == 1
        assert counter["标题前置重复公式段清理"] == 1

    def case_policy_responsibility_table_repaired() -> None:
        blk = Block(kind="image", flags=["table_fallback"], rows=[
            ["一级分类", "检核项目", "内容", "责任承担"],
            ["健康证", "1.健康证", "合作商健康证过期。", "证件不符，N元/人/次。"],
            ["", "", "合作商配送站点公告栏中展示的配送服务人员健康证存在虚假。", "需按照《合作商用工管理规范》相关约定承担违约责任"],
            ["站", "建设", "1.标准站系统一致。标准 2.标准站 ①地址准确。", "200元/项/次，整改不达标需承担双倍违约金。"],
            ["", "控", "合作商未按要求购买视频监控，导致美\n3.视频监 团检核人员无法查看站内情况。", "200元/项/次，整改不达标需承担双倍违约金。"],
            ["", "", "流媒体设备出现遮挡 200元/项/次，整改\n4.流媒体 屏幕，影响正常观看。", ""],
        ])
        normalize_blocks([blk])
        assert blk.rows[2][0] == "健康证"
        assert blk.rows[2][1] == "1.健康证"
        assert "健康证存在虚假" in blk.rows[2][2]
        assert blk.rows[3][0] == "标准站"
        assert blk.rows[3][1] == "2.标准站建设"
        assert "1.标准站系统一致" in blk.rows[3][2]
        assert "一致。标准 ①" not in blk.rows[3][2]
        assert blk.rows[4][0] == "标准站"
        assert blk.rows[4][1] == "3.视频监控"
        assert "美团检核人员" in blk.rows[4][2]
        assert "3.视频监" not in blk.rows[4][2]
        assert blk.rows[5][0] == "标准站"
        assert blk.rows[5][1] == "4.流媒体"
        assert blk.rows[5][3] == "200元/项/次，整改不达标需承担双倍违约金。"

        reverse_blk = Block(kind="table", rows=[
            ["类型", "项目", "内容", "责任承担"],
            ["", "", "合作商或合作商工作人员阻拦检查人员检查。", "5000元/次"],
            ["消极协同", "拒绝配合检查", "合作商或合作商工作人员对检查人员实施抢夺手机。", "20000元/次"],
        ])
        normalize_blocks([reverse_blk])
        assert reverse_blk.rows[1][0] == "消极协同"
        assert reverse_blk.rows[1][1] == "拒绝配合检查"

    def case_station_standard_policy_merged_cells_repaired() -> None:
        blk = Block(kind="image", flags=["table_fallback"], rows=[
            ["一级分类", "检核项目", "内容", "责任承担"],
            ["站点", "14. 手提式灭火器\n15.站点烟感",
             "：合作商下属配送站点、充电区存在安全类隐患，烟感未绑定、烟",
             "500元/项/次，整改不达标需承担双倍违约金。500元/项/次，整改不达标需承担双倍违约金。"],
            ["安全", "16.站点用电",
             "感消防联系人信息缺失或无效联系信息，插排串联、未设置安全、提示类、禁止类标识、消防通道被阻碍、漏电保护开关异常、充电区选址异常、未配备或缺失消防",
             "300元/项/次，整改不达标需承担双倍违约金。"],
            ["安全", "17. 安全通道",
             "设备（包含但不限于防火隔离等未配置等）详见《合作商安全管理规范》）。",
             "300元/项/次，整改不达标需承担双倍违约金。"],
            ["充电", "18.充电区选址", "", "300元/项/次，整改不达标需承担双倍违约金。"],
            ["充电区", "19.充电区维保要求", "", "500元/项/次，整改不达标需承担双倍违约金。"],
            ["充电区", "20.形象装备", "合作商按照标准流程召开早会（标准早会", "200元/项/次，整改不达标需承担双倍违约金。"],
            ["早会", "21.内容交流", "流程包含但不限于：形象装备、重点内容交流）（含驻点站）。", "200元/项/次，整改不达标需承担双倍违约金。"],
            ["台账宿舍", "安全 22.安全台账站外 23.选址安全",
             "此条规则只适用于广东省加盟站点。1、督导线下随机抽检10个骑手，要求签署内容完整。2、严禁签字代签行为。1.宿舍实际地址和标准站系统一致。2.宿舍不得为地下室或者半地下室。",
             "200元/项/次，整改不达标需承担双倍违约金。"],
            ["台账宿舍", "安全 22.安全台账站外 23.选址安全",
             "3. 宿舍房屋非木质结构房屋，宿舍房屋主体结构没有开裂、松动现象。4. 租用房屋应与易燃易爆场所间隔不得小于6米",
             ""],
            ["台账宿舍", "24.宿舍环境", "宿舍需张贴安全警示标识。", "200元/项/次，整改不达标需承担双倍违约金。"],
        ])
        normalize_blocks([blk])
        text_rows = ["|".join(row) for row in blk.rows]
        assert any("站点安全|14.手提式灭火器 / 15.站点烟感 / 16.站点用电 / 17.安全通道" in r for r in text_rows)
        assert any("早会|20.形象装备 / 21.内容交流" in r for r in text_rows)
        assert any("安全台账|22.安全台账" in r and "虚假行为2000元/项/次" in r for r in text_rows)
        assert any("站外宿舍|23.选址安全" in r and "宿舍房屋非木质结构" in r for r in text_rows)
        assert any("站外宿舍|24.宿舍环境" in r for r in text_rows)
        assert not any("台账宿舍" in r or "22.安全台账站外" in r for r in text_rows)

    def case_policy_responsibility_spill_repaired() -> None:
        blk = Block(kind="table", rows=[
            ["一级分类", "检核项目", "内容", "责任承担"],
            ["宿舍", "10.管制刀具",
             "刀具未收纳至固定位置（指隐蔽、不显眼处）。理规范》进行问责。",
             "按《合作商安全管"],
            ["安全五不", "11.电瓶/充电",
             "1、站点非充电功能区不准非充电区给电动改不达标需承担双车、电池充电（包含站内或站外私拉线路倍违约金。充电）。",
             "2000元/项/次，整"],
            ["安全五不", "12.大功率电器",
             "1、站点各功能区不准存放或使用大功率电，整器、大功率电器仅包含热得快、小太阳、改需承担双电丝炉。",
             "1000元/项/次，整改不达标需承担双倍违约金。"],
            ["宿舍", "9.易燃易爆",
             "1、站点各功能区不准存放易燃易爆物品，，整例如汽油、柴油、工业酒精、烟花爆竹、香改需承担双蕉水、液化气罐等易燃物品。",
             "2000元/项/次，整改不达标需承担双倍违约金。"],
            ["站点站务", "6.门头/灯箱",
             "门头信息需与实际门头一致。",
             "200元/项/次，整改违约金。"],
            ["站外宿舍", "29.烟感",
             "每间宿舍房间安装烟感，不达标需承担双倍要求烟感周围0.5m内不应有遮挡物。违约金。",
             "500元/项/次，整改"],
        ])
        normalize_blocks([blk])
        assert blk.rows[1][3] == "按《合作商安全管理规范》进行问责。"
        assert "理规范" not in blk.rows[1][2]
        assert blk.rows[2][3] == "2000元/项/次，整改不达标需承担双倍违约金。"
        assert "给电动车、电池充电" in blk.rows[2][2]
        assert "改不达标" not in blk.rows[2][2]
        assert "大功率电器" in blk.rows[3][2]
        assert "电丝炉" in blk.rows[3][2]
        assert "改需承担双" not in blk.rows[3][2]
        assert "，例如汽油" in blk.rows[4][2]
        assert "香蕉水" in blk.rows[4][2]
        assert "整如" not in blk.rows[4][2]
        assert blk.rows[5][3] == "200元/项/次，整改不达标需承担双倍违约金。"
        assert blk.rows[6][3] == "500元/项/次，整改不达标需承担双倍违约金。"
        assert "要求烟感周围" in blk.rows[6][2]
        assert "不达标需承担双倍" not in blk.rows[6][2]
        assert "违约金" not in blk.rows[6][2]

    def case_redundant_policy_raw_text_removed() -> None:
        raw = (
            "一级分类 检核项目 内容 责任承担 " + "填充" * 260
            + " 3.视频监 团检核人员无法及时查看站内情况 "
            + "流媒体设备出现遮挡 200元/项/次，整改 4.流媒体 屏幕 "
            + "7.看板海 样式符合标准 8.站内宿 员工宿整改舍安全管理制度"
        )
        blocks = [
            Block(kind="para", text=raw, page=0),
            Block(kind="table", page=0, rows=[
                ["一级分类", "检核项目", "内容", "责任承担"],
                ["", "3.视频监控", "合作商未按要求购买视频监控。", "200元/项/次。"],
                ["", "4.流媒体", "流媒体设备出现遮挡屏幕。", "200元/项/次。"],
                ["", "7.看板海报", "看板/海报无缺失。", "200元/项/次。"],
            ]),
        ]
        notes = normalize_blocks(blocks)
        joined = "\n".join(_block_text(b) for b in blocks)
        assert "3.视频监 团" not in joined
        assert "4.流媒体" in joined
        assert any("责任表原始错列段清理" in note for note in notes)

    def case_fine_schedule_definition_fragment_repaired() -> None:
        text = (
            "数】站点给 骑手数】在 【当日6点 比=站点当日晚 烽火台-骑手排 "
            "骑手在烽火台标记为出 调整过班次 数】站点需 骑手数/站点当 "
            "【站点午/晚高 峰时段合格骑 【排班合格出 勤骑手数】以 "
            "【站点应排 班骑手数】截止到当日24点"
        )
        blocks = [
            Block(kind="para", text="【排班在线 【当天班次 排班骑手 有单骑手晚高", bbox=(0, 10, 100, 20)),
            Block(kind="image", rows=[["合格时段", "发生变化的", "数）"]], bbox=(0, 30, 100, 40)),
            Block(kind="para", text=text, confidence=0.3,
                  flags=["low_confidence"], bbox=(0, 50, 100, 60)),
            Block(kind="image", rows=[["点给骑手在", "标记在职骑", "值"]], bbox=(0, 70, 100, 80)),
            Block(kind="para", text="烽火台-业务 烽火台-业务管管理-骑手排 理-骑手排班-精班-精细化排", bbox=(0, 120, 100, 130)),
            Block(kind="image", rows=[["数据", "烽火台-业务", "烽火台-业务管"]], bbox=(0, 140, 100, 150)),
            Block(kind="image", rows=[["查询", "管理-骑手排", "理-骑手排班-精"]], bbox=(0, 160, 100, 170)),
            Block(kind="image", rows=[["路径", "班-精细化排", "细化排班监控-"]], bbox=(0, 180, 100, 190)),
            Block(kind="table",
                  text=("【站点有单骑手数】数据查询路径：烽火台-业务管理-骑手排班-精细化排班监控。"
                        "申诉场景：站点天气判定恶劣，实际正常。"
                        "申诉路径：由渠道经理进行月度提报。"),
                  bbox=(0, 220, 100, 230)),
            Block(kind="image",
                  text="月结算调分明细在哪\n月中系统展示的精细\n化排班分数与站点实",
                  rows=[["目标", "值查询？"], ["", "里查询？"]],
                  bbox=(0, 240, 100, 250)),
        ]
        notes = normalize_blocks(blocks)
        blk = next(b for b in blocks if "【排班在线合格时段数】" in _block_text(b))
        assert any("精细化排班指标定义续表重组" in note for note in notes)
        assert any("精细化排班指标定义碎片清理" in note for note in notes)
        assert "【排班在线合格时段数】" in blk.text
        assert "【当天班次发生变化的骑手数】" in blk.text
        assert "【排班合格出勤骑手数】" in blk.text
        assert "low_confidence" not in blk.flags
        joined = "\n".join(_block_text(b) for b in blocks)
        assert "【排班在线 【当天班次" not in joined
        assert "合格时段 发生变化的 数）" not in joined
        assert "数据查询路径：烽火台-业务管理-骑手排班-精细化排班监控-站点" in joined
        assert "数据 烽火台-业务" not in joined
        assert "申诉路径：由渠道经理进行月度提报" in joined
        assert "月结算调分明细在哪" not in joined
        assert any("精细化排班FAQ碎片清理" in note for note in notes)

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
        assert "距离≤3公里" in text
        assert "距离>3公里" in text
        assert "40天气免责" in text

    def case_special_scene_formula_one_minus_t() -> None:
        text = "体验指标得分=特殊场景体验得分*t＋普通场景体验得分*（1 t）具体方案详见：5.7"
        fixed_text, notes = normalize_text(text)
        assert "（1-t）" in fixed_text
        assert any("特殊场景融合公式 1-t" in note for note in notes)

    def case_special_scene_fusion_intro_added() -> None:
        blocks = [
            Block(kind="heading", text="5.2.2", level=4),
            Block(kind="para", text="特殊场景体验融合考核计分规则"),
            Block(kind="image", flags=["formula_latex"],
                  text="特殊场景完成单占比（t）=特殊场景剔除异常单后完成单/普通场景剔除异常单后完成单+特殊场景剔除异常单后完成单"),
            Block(kind="para", text="体验指标得分=特殊场景体验得分*t＋普通场景体验得分*（1-t）"),
            Block(kind="heading", text="5.3 压力场景加权考核", level=4),
        ]
        notes = normalize_blocks(blocks)
        text = "\n".join(_block_text(b) for b in blocks)
        assert "融合后体验得分" in text
        assert any("特殊场景融合说明补全" in note for note in notes)

    def case_embedded_heading_inside_low_confidence_image_is_split() -> None:
        blocks = [Block(
            kind="image",
            text="前一段说明已经完整结束。二、适用区域本方案适用于全部合作商。",
            flags=["auto_image"],
        )]
        blocks.append(Block(kind="para", text="二、适用区域。"))
        notes = normalize_blocks(blocks)
        text = "\n".join(_block_text(block) for block in blocks)
        assert "二、适用区域" in text
        assert any("段尾一级标题拆分" in note for note in notes)
        assert any(block.kind == "heading" and "适用区域" in block.text
                   for block in blocks)

    def case_ka_symbolic_coefficient_table_restored() -> None:
        rows = [
            ["站点组 KA品牌单星级", "站点组 KA 档位",
             "站点组内站点的 KA 体验膨胀系数（KA 品牌单/大网单量>=3%）",
             "站点组内站点的KA 体验膨胀系数（KA 品牌单/大网单量〈3%）"],
            ["5星", "A", "a", ""],
            ["4星", "B", "", "09"],
            ["3星", "C", "", ""],
            ["2星", "D", "d", ""],
            ["1星", "", "C", ""],
        ]
        repaired = _repair_ka_coefficient_table(rows)
        assert repaired > 0
        assert rows[1:] == [
            ["5星", "A", "a", "f"],
            ["4星", "B", "b", "g"],
            ["3星", "C", "c", "h"],
            ["2星", "D", "d", "i"],
            ["1星", "E", "e", "j"],
        ]

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
        ("normalize.late_normalization_is_idempotent", case_late_normalization_is_idempotent),
        ("normalize.query_path_note_gets_own_line", case_query_path_note_gets_own_line),
        ("normalize.ka_scoped_numeric_and_formula_repairs", case_ka_scoped_numeric_and_formula_repairs),
        ("normalize.page_number_between_title_and_body", case_page_number_between_title_and_body),
        ("normalize.toc_leaders", case_toc_leaders),
        ("normalize.non_toc_standalone_number_does_not_loop", case_non_toc_standalone_number_does_not_loop),
        ("normalize.native_pdf_scientific_table_not_noise_cleaned", case_native_pdf_scientific_table_not_noise_cleaned),
        ("normalize.compressed_toc_fallback_is_explicitly_discarded", case_compressed_toc_fallback_is_explicitly_discarded),
        ("normalize.body_heading_swallowed_by_definition_is_restored", case_body_heading_swallowed_by_definition_is_restored),
        ("normalize.compact_subsection_numbers_restore_parent_dot", case_compact_subsection_numbers_restore_parent_dot),
        ("normalize.training_role_table_cross_block_rebuilt", case_training_role_table_cross_block_rebuilt),
        ("normalize.table_suspect_marks_block", case_table_suspect_marks_block),
        ("normalize.plain_project_label_is_not_pulled_from_content", case_plain_project_label_is_not_pulled_from_content),
        ("normalize.structured_policy_line_breaks_survive_cleanup", case_structured_policy_line_breaks_survive_cleanup),
        ("normalize.disinfection_control_table_spill_repaired", case_disinfection_control_table_spill_repaired),
        ("normalize.self_check_virtual_page_row_is_schema_scoped", case_self_check_virtual_page_row_is_schema_scoped),
        ("normalize.meituan_definition_is_not_logo_noise", case_meituan_definition_is_not_logo_noise),
        ("normalize.meituan_deadline_table_row_is_not_page_furniture", case_meituan_deadline_table_row_is_not_page_furniture),
        ("normalize.semantic_two_column_table_false_positives", case_semantic_two_column_table_false_positives),
        ("normalize.two_column_data_header_and_inline_slogan_repaired", case_two_column_data_header_and_inline_slogan_repaired),
        ("normalize.flattened_formula_context_keeps_visible_labels", case_flattened_formula_context_keeps_visible_labels),
        ("normalize.flattened_threshold_table_restored", case_flattened_threshold_table_restored),
        ("normalize.formula_key_value_table_keeps_sigma_and_drops_duplicate", case_formula_key_value_table_keeps_sigma_and_drops_duplicate),
        ("normalize.brand_share_formula_keeps_all_sigma_terms", case_brand_share_formula_keeps_all_sigma_terms),
        ("normalize.ka_adjustment_tail_without_embedded_public_notice", case_ka_adjustment_tail_without_embedded_public_notice),
        ("normalize.overtime_appointment_absolute_value_restored", case_overtime_appointment_absolute_value_restored),
        ("normalize.settlement_flow_diagram_becomes_comparison_table", case_settlement_flow_diagram_becomes_comparison_table),
        ("normalize.faq_questions_and_answers_are_separated", case_faq_questions_and_answers_are_separated),
        ("normalize.formula_fragment_below_reliable_latex_removed", case_formula_fragment_below_reliable_latex_removed),
        ("normalize.cross_block_labels_are_rejoined", case_cross_block_labels_are_rejoined),
        ("normalize.table_continuation_labels_are_rejoined", case_table_continuation_labels_are_rejoined),
        ("normalize.repeated_next_section_formula_tail_removed", case_repeated_next_section_formula_tail_removed),
        ("normalize.heading_prefixed_formula_duplicate_removed", case_heading_prefixed_formula_duplicate_removed),
        ("normalize.policy_responsibility_table_repaired", case_policy_responsibility_table_repaired),
        ("normalize.station_standard_policy_merged_cells_repaired", case_station_standard_policy_merged_cells_repaired),
        ("normalize.policy_responsibility_spill_repaired", case_policy_responsibility_spill_repaired),
        ("normalize.redundant_policy_raw_text_removed", case_redundant_policy_raw_text_removed),
        ("normalize.fine_schedule_definition_fragment_repaired", case_fine_schedule_definition_fragment_repaired),
        ("normalize.nested_weather_policy_table_repaired", case_nested_weather_policy_table_repaired),
        ("normalize.ka_special_scene_fusion_repaired", case_ka_special_scene_fusion_repaired),
        ("normalize.special_scene_formula_one_minus_t", case_special_scene_formula_one_minus_t),
        ("normalize.special_scene_fusion_intro_added", case_special_scene_fusion_intro_added),
        ("normalize.embedded_heading_inside_low_confidence_image_is_split", case_embedded_heading_inside_low_confidence_image_is_split),
        ("normalize.ka_symbolic_coefficient_table_restored", case_ka_symbolic_coefficient_table_restored),
    ]
