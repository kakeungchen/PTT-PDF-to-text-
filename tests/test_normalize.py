import unittest

from ptt.models import Block
from ptt.normalize import (clean_table_noise_rows, is_noise_text,
                           normalize_blocks, normalize_text,
                           table_suspect_score)


def _test_block_text(block):
    parts = [block.text or ""]
    if block.rows:
        parts.extend(" ".join(c for c in row if c) for row in block.rows)
    return "\n".join(p for p in parts if p)


class NormalizeTextTests(unittest.TestCase):
    def fixed(self, text):
        return normalize_text(text)[0]

    def test_sigma_formula_context(self):
        text = "站点标准化率=二检核项得分/工检核项配分"
        self.assertEqual(self.fixed(text), "站点标准化率=Σ 检核项得分/Σ 检核项配分")

    def test_sigma_missing_around_rate_formula(self):
        text = "早会标准化率=（日常早会通过天数）/有单天数；消毒标准化率=（通过人数）/工有单天数"
        self.assertEqual(
            self.fixed(text),
            "早会标准化率=Σ（日常早会通过天数）/Σ 有单天数；消毒标准化率=Σ（通过人数）/Σ 有单天数",
        )

    def test_sigma_natural_week_context(self):
        text = "站点有效骑手数：二自然周站点有单天数大于2的骑手数"
        self.assertEqual(self.fixed(text), "站点有效骑手数：Σ 自然周站点有单天数大于2的骑手数")

    def test_amount_and_digit_spacing(self):
        text = "N=3 00，N= 400，10:00至2 0:00，可视区域达到 8 0%，200元/项//次，i200元/项/饮"
        self.assertEqual(
            self.fixed(text),
            "N=300，N=400，10:00至20:00，可视区域达到 80%，200元/项/次，200元/项/次",
        )

    def test_common_ocr_confusions(self):
        text = "提备路径，墻体，视沩满足，超时将无法中诉，双方因予遵守，督导大区负责人一—yan1i03"
        self.assertEqual(
            self.fixed(text),
            "提报路径，墙体，视为满足，超时将无法申诉，双方应予遵守，督导大区负责人——yanli03",
        )

    def test_station_id_rule_does_not_rewrite_valid_id_spacing(self):
        text = "场所格式力“站点 ID+站点名称”，门店编号格式为“站点1习D”"
        self.assertEqual(
            self.fixed(text),
            "场所格式为“站点 ID+站点名称”，门店编号格式为“站点ID”",
        )

    def test_more_feedback_confusions(self):
        text = "任一项未达标，该项结果沩驳回；填写岗位的人员信息非本人或己离职；站点I D；LOG0不清；违规违约甲诉，不子申诉支持"
        self.assertEqual(
            self.fixed(text),
            "任一项未达标，该项结果为驳回；填写岗位的人员信息非本人或已离职；站点ID；LOGO不清；违规违约申诉，不予申诉支持",
        )

    def test_interview_policy_typo_is_fixed(self):
        self.assertEqual(self.fixed("重度警告讳约选站标准"), "重度警告违约选站标准")

    def test_score_phrase_confusions(self):
        text = "得分力0分；得分力-1；得分 - 1；得分1；得分为- 0.1；兜底力0分；以实际签署准"
        self.assertEqual(
            self.fixed(text),
            "得分为0分；得分为-1；得分为-1；得分为1；得分为-0.1；兜底为0分；以实际签署为准",
        )

    def test_salary_policy_confusions(self):
        self.assertEqual(normalize_text("0=", "table")[0], "=0")
        self.assertEqual(
            self.fixed("留存目标达成率=（有效骑手留存率/有效骑手留存率目标）×100%备注"),
            "留存目标达成率=（有效骑手留存率/有效骑手留存率目标）×100%。备注",
        )
        text = (
            "天气等级力10，天气指数力10，站点组剔除异常单后白，"
            "普通场景和特场景融合，烽火台-商服务费考核方案，"
            "有效骑手留存分母备注：分子分母权重力2，"
            "留存日标达成率=（有效骑手留存率/有效骑手留存率目标） 100%备注，"
            "分数计算规 分数区间：［-0.5，+0.5］则 留存率达标线"
        )
        self.assertEqual(
            self.fixed(text),
            "天气等级为10，天气指数为10，站点组剔除异常单后的，"
            "普通场景和特殊场景融合，烽火台-商户服务费考核方案，"
            "有效骑手留存分母。备注：分子分母权重为2，"
            "留存目标达成率=（有效骑手留存率/有效骑手留存率目标）×100%。备注，"
            "分数计算规则：分数区间：［-0.5，+0.5］。留存率达标线",
        )
        self.assertEqual(self.fixed("特场景品牌订单"), "特殊场景品牌订单")

    def test_inline_numbered_clauses_are_split(self):
        text = "双方应予遵守。2. 本方案于2026年6月1日生效。"
        self.assertEqual(
            self.fixed(text),
            "双方应予遵守。\n\n2. 本方案于2026年6月1日生效。",
        )

    def test_more_ka_feedback_confusions(self):
        text = "展示沩大网单，结算方窝，站点姐A，强排抯，膨胀系数力，合作商签暑，肯得牌"
        self.assertEqual(
            self.fixed(text),
            "展示为大网单，结算方案，站点组A，强排组，膨胀系数为，合作商签署，肯德基品牌",
        )

    def test_image_caption_feedback_confusions(self):
        text = (
            "加盟站点 AI、A3 和集约站点 A2. A4；"
            "《KA品牌肇月度 强排擅 站点组强推 集圴站 KA群合考核 KA品牌单 汁算 "
            "（则除异常单） 也会参与《专送合作商服务类结算方案）。"
        )
        self.assertEqual(
            self.fixed(text),
            "加盟站点 A1、A3 和集约站点 A2、A4；"
            "《KA品牌单月度 强排组 站点组强排 集约站 KA融合考核 KA品牌单 计算 "
            "（剔除异常单） 也会参与《专送合作商服务类结算方案》。",
        )

    def test_formula_variable_sequence_confusions(self):
        text = "各 KA品牌单量 Ki、Kz. Ks.. Ka；各KA品牌体验得分 Ki分、Ka分、Ks分..Kx分；各KA 品牌权重 Q1 .Q2.Q3..Qx"
        self.assertEqual(
            self.fixed(text),
            "各 KA品牌单量 K1、K2、K3...Kx；各KA品牌体验得分 K1分、K2分、K3分...Kx分；各KA 品牌权重 Q1、Q2、Q3...Qx",
        )

    def test_page_number_between_title_and_body(self):
        text = "2026年6月薪动力专送合作商站点星级考核制度-KA 品牌运力调分补充协议 2本协议内容为"
        self.assertEqual(
            self.fixed(text),
            "2026年6月薪动力专送合作商站点星级考核制度-KA 品牌运力调分补充协议\n\n本协议内容为",
        )

    def test_toc_leaders(self):
        text = "一、 总则 ⋯ ⋯19"
        self.assertEqual(normalize_text(text, "toc")[0], "一、 总则　19")

    def test_numbered_heading_comma_noise(self):
        text = "四、， 服务费结算方案⋯ 2"
        self.assertEqual(self.fixed(text), "四、服务费结算方案　2")

    def test_mixed_toc_page_marker_noise(self):
        text = "一、背景.. ：1"
        self.assertEqual(self.fixed(text), "一、背景　1")


class NormalizeBlockTests(unittest.TestCase):
    def test_table_suspect_marks_block(self):
        blk = Block(kind="table", rows=[
            ["项目", "内容", "责任承担"],
            ["4.站点经营 票等。", "1、严禁存放非美团装备。200元/项/次，整改不达标", "达标需承担双倍违约金。"],
        ])
        notes = normalize_blocks([blk])
        self.assertIn("table_low_confidence", blk.flags)
        self.assertGreaterEqual(table_suspect_score(blk.rows), 3)
        self.assertIsInstance(notes, list)

    def test_merged_table_headers_are_suspicious(self):
        rows = [
            ["项目", "内容", "说明 承担责任"],
            ["基础检核", "日常提交", "提交系统 500元/次"],
        ]
        self.assertGreaterEqual(table_suspect_score(rows), 3)

    def test_one_column_swallowed_table_is_suspicious(self):
        rows = [[
            "类型 项目 内容 责任承担 合作商工作人员需配备头盔正版服装外卖箱。"
            "安全头盔检核不合规200元/人/次，服装检核不合规100元/人/次，"
            "餐箱检核不合规100元/人/次，承担违约金。"
        ]]
        self.assertGreaterEqual(table_suspect_score(rows), 3)

    def test_long_garbled_table_row_is_suspicious(self):
        rows = [[
            "补充说明",
            "1 参与考核站点本制度则按照2 同商同配送区约站、加盟大/是否属于同配\n"
            "③ 考核品牌门店基（含肯德基门店之外的全1 单量门槛：考于300单的不虚假单、提前组承接的必胜客调分项考核\n"
            "⑤ 数据获取方式3 该调分项，只",
            "组：以实际签署为准，若考核月新建集约站本制度进行考核，若未签署则不参与本制度或的加盟站点作为站点组参与考核，"
            "考核数x站，不含企客集约站（即不含站点名称含“送区同商情况以考核周期最后一天状态为准：麦当劳品牌的驻点和融网门店"
            "（剔除品牌肯悦咖啡\\百胜轻食\\爷爷自在茶等全部子品部门店）、必胜客（剔除品牌方不考核门店亥月站点组承接的麦当劳完单量",
            "点归属的合作商签暑考核；据合并计算，含KA導企客集约”的站点）不考核门店）；肯得牌，除品牌方不考材包、提前关闭订单），"
            "的肯德基单量（剔B项考核，考核月站，300单的不参与必月",
        ]]
        self.assertGreaterEqual(table_suspect_score(rows), 3)

    def test_malformed_percent_cell_is_suspicious(self):
        rows = [
            ["要求", "品牌方口径准时率", "考核周期得分"],
            ["目标值", "%66", "0.2"],
        ]
        self.assertGreaterEqual(table_suspect_score(rows), 3)

    def test_broken_formula_variable_table_is_suspicious(self):
        rows = [
            ["参考指标", "代表字母"],
            ["各KA 品牌单量", "K..Ka. Ks..Kx"],
            ["各KA 品牌体验得分", "KI分、Ka分、Ks分KN分"],
            ["各 KA品牌权重", "Q1.Q2"],
        ]
        self.assertGreaterEqual(table_suspect_score(rows), 3)

    def test_nested_weather_policy_table_is_suspicious(self):
        rows = [
            ["方案", "", "", "内容", ""],
            ["考核规则", "以运单首次调度时的天", "等级为依据，按", "下方规则分普通场景和", "特殊场景考核并计算得分："],
            ["", "指标", "普通场景考核", "特殊场景考核", "备注"],
            ["", "负向反馈率", "天气等级为10", "导航距离>3 公里", ""],
            ["", "配送原因未完成率", "且导航距离≤3公里", "的正常天气单恶劣天气单（天气等级 20或30或", "40天气免责"],
            ["", "复合准时率（考核）", "", "40）HD 尾单&专送兜底", "40天气或导航距离>5公里免责"],
            ["", "复合超时时长", "", "单", "40天气或导航距离>5公里免责"],
        ]
        self.assertGreaterEqual(table_suspect_score(rows), 3)

    def test_nested_weather_policy_table_is_repaired(self):
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

        self.assertEqual(blk.rows[0], ["指标", "普通场景考核", "特殊场景考核", "备注"])
        self.assertEqual(blk.rows[1][0], "负向反馈率")
        self.assertIn("天气等级为10", blk.rows[1][1])
        self.assertIn("HD尾单&专送兜底单", blk.rows[1][2])
        self.assertEqual(blk.rows[2][3], "40天气免责")

    def test_split_total_completed_order_cell_is_repaired(self):
        blk = Block(kind="table", rows=[
            ["参考指标", "超时区间", "超时时长计算逻辑"],
            ["总完成单（W）", "站点组剔除异常单后白", "总完成单"],
        ])
        notes = normalize_blocks([blk])

        self.assertEqual(
            blk.rows[1],
            ["总完成单（W）", "站点组剔除异常单后的总完成单", ""],
        )
        self.assertTrue(any("表格单元格错拆修复" in note for note in notes))

    def test_ka_coefficient_table_rows_are_repaired(self):
        blk = Block(kind="table", rows=[
            ["站点组KA", "站点组KA", "站点组内站点的KA 体验膨胀系数", "站点组内站点的KA 体验膨胀系数"],
            ["星级", "档位", "（KA 品牌单/大网单量>=3%）", "（KA 品牌单/大网单量〈3%）"],
            ["5星4星", "A B", "1.0151.015", "1.00751.0075"],
            ["3星", "C", "1.005", "1.0025"],
            ["2星", "D", "", ""],
            ["1星", "", "0.92", "0.96"],
        ])

        normalize_blocks([blk])

        self.assertEqual(blk.rows, [
            ["站点组KA星级", "站点组KA档位",
             "站点组内站点的KA 体验膨胀系数（KA 品牌单/大网单量>=3%）",
             "站点组内站点的KA 体验膨胀系数（KA 品牌单/大网单量<3%）"],
            ["5星", "A", "1.015", "1.0075"],
            ["4星", "B", "1.015", "1.0075"],
            ["3星", "C", "1.005", "1.0025"],
            ["2星", "D", "1", "1"],
            ["1星", "E", "0.92", "0.96"],
        ])

    def test_ka_city_coefficient_table_missing_grades_are_repaired(self):
        blk = Block(kind="table", rows=[
            ["站点组KA", "站点组KA", "站点组内站点的KA 体验膨胀系数", "站点组内站点的KA 体验膨胀系数"],
            ["星级", "档位", "（KA 品牌单/大网单量>=3%）", "（KA 品牌单/大网单量〈3%）"],
            ["5星", "A", "1.03", "1.015"],
            ["4星", "", "1.03", "1.015"],
            ["3星", "C", "1.01", "1.005"],
            ["2星", "D", "", ""],
            ["1星", "", "0.84", "0.92"],
        ])

        normalize_blocks([blk])

        self.assertEqual(blk.rows[2], ["4星", "B", "1.03", "1.015"])
        self.assertEqual(blk.rows[4], ["2星", "D", "1", "1"])
        self.assertEqual(blk.rows[5], ["1星", "E", "0.84", "0.92"])

    def test_ka_variable_sequences_are_normalized(self):
        text, _ = normalize_text("Q1 .Q2.93QN / Ki.Ke. Ks..KN / K1分、K2分、Kg分..KN分", "table")

        self.assertIn("Q1、Q2、Q3...Qn", text)
        self.assertIn("K1、K2、K3...Kx", text)
        self.assertIn("K1分、K2分、K3分...Kx分", text)

    def test_bishengke_target_percent_is_repaired(self):
        blk = Block(kind="table", rows=[
            ["要求", "品牌方口径准时率（必胜客）", "考核周期得分"],
            ["目标值", "%66", "0.2"],
        ])

        normalize_blocks([blk])

        self.assertEqual(blk.rows[1][1], "99%")

    def test_flattened_kfc_punctuality_table_is_repaired(self):
        blk = Block(kind="table", rows=[
            ["要求目标值介于门槛值、目标值之间",
             "品牌方口径准时率（肯德基）99%介于93%到 99%之间",
             "考核周期得分为0.2等比例计算得分"],
            ["门槛值", "93%", "0"],
        ])

        normalize_blocks([blk])

        self.assertEqual(blk.rows, [
            ["要求", "品牌方口径准时率（肯德基）", "考核周期得分"],
            ["目标值", "99%", "0.2"],
            ["介于门槛值、目标值之间", "介于93%到99%之间", "等比例计算得分"],
            ["门槛值", "93%", "0"],
        ])

    def test_ka_assessment_framework_diagram_is_repaired(self):
        blk = Block(kind="image", flags=["table_fallback"], rows=[
            ["考核方案", "常规计分项", "调分项", "调星项"],
            ["", "①复合准时率 ②配送原因未完成率 ③KA品牌负向反馈率",
             "①KA品牌驻点骑手达标率", "①【客诉虚假点送达】降星"],
            ["", "④KA品牌客诉率（瑞幸）⑤承托比（星巴克）⑥虚假点送达率 ⑦复合超时时长",
             "②KA品牌体验调分项", "②【B端客诉事件】降星 ③【履约原因门店流失】降星"],
        ])

        normalize_blocks([blk])

        self.assertEqual(blk.rows[0], ["模块", "类别", "内容"])
        self.assertEqual(blk.rows[1][1], "常规计分项")
        self.assertIn("⑦ 复合超时时长", blk.rows[1][2])
        self.assertIn("只影响KA星级", blk.rows[2][2])
        self.assertIn("履约原因门店流失", blk.rows[3][2])

    def test_fragmented_ka_assessment_framework_is_rebuilt(self):
        blocks = [
            Block(kind="heading", text="二、 考核周期", level=2),
            Block(kind="para", text="2026年6月1日至2026年6月30日。"),
            Block(kind="table", rows=[["考核方案"]]),
            Block(kind="table", rows=[["常规计分项", "调分项", "调星项"]]),
            Block(kind="para", text="①复合准时率 ①KA 品牌驻点骑手达标率（只影 ① 【客诉虚假点送达】降星"),
            Block(kind="table", rows=[["②配送原因未完成率", "响KA 星级）", "【B端客诉事件】降星"]]),
            Block(kind="para", text="③KA 品牌负向反馈率 ②KA 品牌体验调分项（只影响站点组星级）③【履约原因门店流失】降星"),
            Block(kind="para", text="④KA 品牌客诉率（瑞幸）⑤承托比（星巴克）⑥虚假点送达率⑦复合超时时长"),
            Block(kind="heading", text="四、考核细则", level=2),
        ]

        normalize_blocks(blocks)

        self.assertEqual(blocks[2].text, "三、考核框架")
        self.assertEqual(blocks[3].kind, "table")
        self.assertEqual(blocks[3].rows[1][1], "常规计分项")
        self.assertEqual(blocks[4].text, "四、考核细则")
        self.assertIn("只影响KA星级", "\n".join(_test_block_text(b) for b in blocks))

    def test_scene_experience_framework_diagram_is_repaired(self):
        blk = Block(kind="image", flags=["table_fallback"], rows=[
            ["分场景体验考核", "普通场景目标", "特殊场景目标"],
            ["", "正常天气非指定考核日 距离≤3公里单 1倍权重",
             "谢味恕订单 多倍权甫"],
            ["", "正常天气指定考核日（特妹节假日） 多倍权甫",
             "正常天气非指定考核日 >3公里单 1倍权重"],
            ["", "正常天气指定考核日（其他指定日） 多倍权甫",
             "正常天气指定考核日（其他指定日） ≥3公里单 多倍权甫"],
        ])

        normalize_blocks([blk])

        flat = "\n".join("\t".join(r) for r in blk.rows)
        self.assertEqual(blk.rows[0], ["场景归属", "日期/天气条件", "运单范围", "权重"])
        self.assertIn("恶劣天气订单", flat)
        self.assertIn("正常天气指定考核日（特殊节假日）", flat)
        self.assertNotRegex(flat, r"谢味恕|特妹|权甫")

    def test_fragmented_scene_experience_framework_is_rebuilt(self):
        blocks = [
            Block(kind="heading", text="5.1分场景体验考核框架", level=3),
            Block(kind="para", text="分场景体验考核"),
            Block(kind="image", flags=["formula"],
                  text="正常天气非指定考核日 距离≤3公里单 1倍权 谢味恕劣天气订单 多倍权重 特妹场景目标 正常天气指定考核日（特珠节假日） >3公里单 多倍权甫 正常天气指定考核日（其他指定日） ≥3公里单"),
            Block(kind="heading", text="5.2.1 场景说明", level=4),
        ]

        normalize_blocks(blocks)

        self.assertEqual(blocks[1].kind, "table")
        self.assertEqual(blocks[1].rows[0], ["场景归属", "日期/天气条件", "运单范围", "权重"])
        self.assertIn("恶劣天气订单", "\n".join("\t".join(r) for r in blocks[1].rows))
        self.assertEqual(blocks[2].text, "5.2.1 场景说明")

    def test_pressure_scenario_table_repeats_merged_cells(self):
        blk = Block(kind="table", rows=[
            ["压力场景类型", "场景定义", "单量加权系数", "补充说明"],
            ["恶劣天气", "恶劣天气调度的运单，即调度时天气等级>10的运单",
             "X倍（详见烽火台）", "台风、内涝、结冰等3类极端恶劣天气调度的运单可申请异常单剔除。"],
            ["指定考核日", "特殊节假日调度的运单，含法定节假日或其他指定日期内调度的运单。",
             "Y倍（详见烽火台）", "见每月指定考核日期"],
            ["", "周末/周中指定日期调度的运单", "Z倍（详见烽火台）", ""],
        ])

        normalize_blocks([blk])

        self.assertEqual(blk.rows[3][0], "指定考核日")
        self.assertEqual(blk.rows[3][3], "见每月指定考核日期")
        self.assertEqual(len(blk.rows), 4)

    def test_ka_indicator_definition_fills_completed_order_definition(self):
        blk = Block(kind="table", rows=[
            ["参考指标", "代表字母", "统计口径"],
            ["完成订单量（W）", "W_KA品牌单", ""],
            ["一般超时单量", "C_KA品牌单", "KA品牌单中一般超时订单数量"],
            ["严重超时订单量", "Y_KA品牌单", "KA品牌单中严重超时订单数量"],
            ["复合准时率（考核）", "1-(C+5Y)/W", ""],
        ])

        normalize_blocks([blk])

        self.assertEqual(
            blk.rows[1][2],
            "通过美团配送且运单最终状态为完成的运单数量",
        )

    def test_more_ka_indicator_w_definitions_are_filled(self):
        starbucks = Block(kind="table", rows=[
            ["参考指标", "释义"],
            ["打标骑手完成的星巴克单数（R）", "打标骑手配送且运单最终状态为完成的星巴克运单数量"],
            ["完成星巴克订单量（W）", ""],
        ])
        fake_delivery = Block(kind="table", rows=[
            ["参考指标", "释义"],
            ["虚假点送达（复合）订单量（T）", "配送人员虚假点击送达的订单量"],
            ["完成订单量（W）", ""],
        ])

        normalize_blocks([starbucks, fake_delivery])

        self.assertEqual(
            starbucks.rows[2][2],
            "通过美团配送且运单最终状态为完成的星巴克运单数量",
        )
        self.assertEqual(
            fake_delivery.rows[2][2],
            "通过美团配送且运单最终状态为完成的运单数量",
        )

    def test_flattened_ka_scene_policy_table_is_repaired(self):
        blk = Block(kind="table", rows=[
            ["方案", "内容"],
            ["考核方式", "普通场景和特殊场景分两套目标考核，按特殊场景完成单占比计算融合后体验得分。"],
            ["考核指标", "虚假点送达率、配送原因未完成率、复合准时率、KA 品牌负向反馈率、承托比、复合超时时长、KA 品牌客诉率。"],
            ["天气等级", "10：正常天气；20：一般恶劣天气：30：比较恶劣天气：40：非常恶劣天气"],
            ["考核规则", "体验指标 普通场景考核 特殊场景考核 备注"],
            ["", "配送原因未完成率 距离≤3公里的正 距离>3公里的正常 40天气或240天气常天气单（天气等 天气单 免责级力 10） 恶劣天气单（天气等级 20或30）"],
            ["", "复合超时时长 常天气单（天气等 天气单 免责或大于5公里级为 10） 恶劣天气单（天气等 远距离单"],
        ])

        normalize_blocks([blk])

        flat = "\n".join("\t".join(r) for r in blk.rows)
        self.assertEqual(blk.rows[0], ["类型", "项目/指标", "普通场景考核", "特殊场景考核", "备注"])
        self.assertIn("KA品牌客诉率", flat)
        self.assertIn("复合超时时长", flat)
        self.assertIn("40天气免责或大于5公里远距离单", flat)
        self.assertNotIn("级力", flat)
        self.assertNotIn("240天气", flat)

    def test_flattened_ka_score_rule_table_is_repaired(self):
        blk = Block(kind="table", rows=[
            ["", "以复合准时率为例一计分规则："],
            ["复合准时率/", "个 当复合准时率≤X 时，站点得分为0；"],
            ["承托比", "当复合准时率=Y时，站点得分为100分；"],
            ["", "今当复合准时率≥Z时，站点得分为120分；"],
            ["② 负向指标计分规则", "当复合准时率介于（XI,Yi）或（Y,Z.）之间时，等比例算分。"],
            ["配送原因未完成率/", "以配送原因未完成率为例一计分规则："],
            ["KA 品牌负向反馈率/ KA", "当配送原因未完成率≥X时，站点得分为0分；"],
            ["品牌客诉率/虚假点送", "当配送原因未完成率=Y2时，站点得分为100分；"],
            ["达率/复合超时时长", "当配送原因未完成率≤22时，站点得分为120分；"],
            ["", "个 当配送原因未完成率介于（Xe,Yz）或（Yz,Zs）之间时，等比例算分。"],
        ])

        normalize_blocks([blk])

        flat = "\n".join("\t".join(r) for r in blk.rows)
        self.assertEqual(blk.rows[0], ["规则类型", "适用指标", "计分规则"])
        self.assertIn("复合准时率、承托比", flat)
        self.assertIn("配送原因未完成率、KA品牌负向反馈率", flat)
        self.assertIn("当指标≤Z时，站点得分为120分", flat)
        self.assertNotRegex(flat, r"XI|Yi|Y2|≤22|Xe|Yz|Zs")

    def test_ka_indicator_formulas_are_inserted_when_ocr_drops_them(self):
        blocks = [
            Block(kind="heading", text="5.4.1 复合准时率", level=4),
            Block(kind="para", text="计算口径："),
            Block(kind="para", text="WKA品牌单指标释义："),
            Block(kind="table", rows=[
                ["参考指标", "释义"],
                ["一般超时单量（C）", "普通单：8min<送达时间-期待送达时间≤30min"],
                ["严重超时订单量（Y）", "普通单：30min<送达时间-期待送达时间"],
                ["完成订单量（W）", ""],
            ]),
            Block(kind="heading", text="5.4.2 配送原因未完成率", level=4),
            Block(kind="para", text="指标释义："),
            Block(kind="table", rows=[
                ["参考指标", "释义"],
                ["配送原因未完成单量（P）", "因配送原因运单最终状态为取消的运单数量"],
                ["完成订单量（W）", ""],
            ]),
            Block(kind="heading", text="5.4.3 KA 品牌负向反馈率", level=4),
        ]

        normalize_blocks(blocks)

        texts = [_test_block_text(b) for b in blocks]
        self.assertIn("复合准时率（考核）=1-(C_{KA品牌单}+5*Y_{KA品牌单})/W_{KA品牌单}", texts)
        self.assertIn("配送原因未完成率=P_{KA品牌单}/(W_{KA品牌单}+P_{KA品牌单})", texts)
        self.assertIn("计算口径：", texts)
        self.assertIn("指标释义：", texts)
        self.assertTrue(any(
            b.rows and b.rows[-1][-1] == "通过美团配送且运单最终状态为完成的运单数量"
            for b in blocks
        ))

    def test_feedback_city_typo_is_normalized(self):
        self.assertEqual(normalize_text("张冢口站点组", "text")[0], "张家口站点组")

    def test_ka_overtime_formula_is_inserted_and_broken_row_removed(self):
        blocks = [
            Block(kind="heading", text="5.4.7 复合超时时长", level=4),
            Block(kind="table", rows=[
                ["订单类型", "复合超时时长订单口径"],
                ["KA品牌单", "见下表"],
            ]),
            Block(kind="table", rows=[
                ["", "", "WKA品陳車"],
                ["超时类型", "超时区间", "超时时长计算逻辑（秒）"],
                ["A1", "8min<送达时间-期待送达时间≤30min", "送达时间-期待送达时间-8min"],
            ]),
            Block(kind="heading", text="5.4.8 其他指标", level=4),
        ]

        notes = normalize_blocks(blocks)

        texts = [_test_block_text(b) for b in blocks]
        self.assertIn(
            "复合超时时长=(A1_{KA品牌单}+A2_{KA品牌单}+A3_{KA品牌单})/W_{KA品牌单}",
            texts,
        )
        flat = "\n".join(texts)
        self.assertNotIn("WKA品陳車", flat)
        self.assertTrue(any("复合超时时长公式补全" in note for note in notes))

    def test_ka_overtime_example_table_is_relaid_out(self):
        blocks = [
            Block(kind="heading", text="5.4.7 复合超时时长", level=4),
            Block(kind="table", rows=[
                ["例：呆站后", "A，在晋週场芳", "： 共有6笔起", "时 早（非", "定考核日早）"],
                ["后总完成单", "为1000单（考", "虑压力场景下", "单量加权系", "女）。"],
                ["订单编号", "超时时长", "是否异常单", "超时类型", "考核超时时长"],
                ["订单1", "2分钟（120秒）", "非异常单", "不考核", "0秒"],
                ["", "复合", "超时时长=（2", "40+1,320+4，", "170+8,970）/1000=14.7秒"],
            ]),
            Block(kind="heading", text="5.5 数据查询路径及申诉", level=4),
        ]

        notes = normalize_blocks(blocks)

        texts = [_test_block_text(b) for b in blocks]
        flat = "\n".join(texts)
        self.assertIn("示例：某站点A，在普通场景下共有6笔超时订单", flat)
        self.assertIn("复合超时时长=（240+1,320+4,170+8,970）/1000=14.7秒", flat)
        clean_table = next(b for b in blocks if b.kind == "table" and b.rows and b.rows[0][0] == "订单编号")
        self.assertEqual(clean_table.rows[0], ["订单编号", "超时时长", "是否异常单", "超时类型", "考核超时时长"])
        self.assertNotIn("呆站后", flat)
        self.assertNotIn("40+1,320+4，", flat)
        self.assertTrue(any("复合超时时长示例表重排" in note for note in notes))

    def test_ka_score_threshold_tables_are_split(self):
        blocks = [
            Block(kind="para", text="所有品牌按照品牌维度汇总站点组下该品牌所有单量计算得分。"),
            Block(kind="table", rows=[
                ["配送原因未完成率", "得分"],
                ["X2", "0"],
                ["", "115"],
                ["Z2", "120"],
                ["承托比", "得分"],
                ["", "0"],
                ["", "100"],
                ["", "120"],
                ["KA 品牌客诉率", "得分"],
                ["", "0"],
                ["Ys", "115"],
                ["Z8", "120"],
            ]),
            Block(kind="para", text="虚假点送达率 得分Xs 0 Ys 115 Z5120"),
            Block(kind="table", rows=[
                ["复合超时时长", "得分"],
                ["XT", "0"],
                ["", "100"],
                ["Z：", "120"],
            ]),
            Block(kind="para", text="说明（仅为示例，最终以美团配送烽火台-商服务费计费系统为准）："),
        ]

        notes = normalize_blocks(blocks)

        metric_tables = [b for b in blocks if b.kind == "table" and b.rows and b.rows[0][1] == "得分"]
        self.assertEqual(len(metric_tables), 7)
        self.assertEqual(metric_tables[0].rows, [["复合准时率", "得分"], ["X1", "0"], ["Y1", "100"], ["Z1", "120"]])
        self.assertEqual(metric_tables[5].rows, [["KA品牌客诉率", "得分"], ["X8", "0"], ["Y8", "115"], ["Z8", "120"]])
        self.assertEqual(metric_tables[6].rows, [["复合超时时长", "得分"], ["X7", "0"], ["Y7", "100"], ["Z7", "120"]])
        flat = "\n".join(_test_block_text(b) for b in blocks)
        self.assertNotIn("Z5120", flat)
        self.assertNotIn("XT", flat)
        self.assertTrue(any("KA指标阈值表拆分重排" in note for note in notes))

    def test_ka_rider_score_rule_table_is_split_and_relaid_out(self):
        blocks = [
            Block(kind="heading", text="6.1 KA 品牌驻点骑手达标率", level=4),
            Block(kind="table", rows=[
                ["", "【KA 品牌驻点骑手】目标骑手达成天数/要求 考核周期得分该考核周期总天数"],
                ["", "目标值 100% 0"],
                ["", "介于门槛值、 介于 0%到100%之间 等比例计算得分目标值之间"],
                ["", "门槛值 0%"],
                ["申诉和剔除", "站点组可针对下述情景进行运力申诉，每月具体申诉要求可向渠道经理咨询"],
                ["申诉流程", "每月倒数第二个工作日前将需申诉的事项上报至渠道经理"],
                ["运力验证", "微笑行动抽查不合格非本人"],
                ["补充说明", "是否属于同配送区同商情况以考核周期最后一天状态准。"],
            ]),
            Block(kind="heading", text="7. 站点组星级调分项", level=3),
        ]

        notes = normalize_blocks(blocks)

        tables = [b for b in blocks if b.kind == "table" and b.rows]
        self.assertEqual(tables[0].rows[0], ["要求", "【KA品牌驻点骑手】目标骑手达成天数/考核周期总天数", "考核周期得分"])
        self.assertEqual(tables[0].rows[-1], ["门槛值", "0%", "-3"])
        self.assertEqual(tables[1].rows[0], ["项目", "内容"])
        self.assertIn("新商换站；2、极端恶劣天气", tables[1].rows[1][1])
        self.assertIn("状态为准", tables[1].rows[-1][1])
        flat = "\n".join(_test_block_text(b) for b in blocks)
        self.assertNotIn("目标骑手达成天数/要求 考核周期得分", flat)
        self.assertTrue(any("KA驻点骑手计分表重排" in note for note in notes))

    def test_ka_experience_adjustment_missing_definition_is_inserted(self):
        blocks = [
            Block(kind="heading", text="7.1 KA 品牌体验调分项", level=4),
            Block(kind="para", text="计分规则1：麦当劳30分钟内送达订单占比。"),
            Block(kind="heading", text="7.2 KA品牌驻点骑手达标率", level=4),
        ]

        notes = normalize_blocks(blocks)

        table = next(b for b in blocks if b.kind == "table" and b.rows)
        flat = "\n".join("\t".join(r) for r in table.rows)
        self.assertIn("得分范围", flat)
        self.assertIn("麦当劳完单量占比", flat)
        self.assertIn("品牌方口径准时率（肯德基）", flat)
        self.assertIn("麦肯必单量占比", flat)
        self.assertTrue(any("KA品牌体验调分项定义补全" in note for note in notes))

    def test_ka_fake_delivery_missing_definition_is_inserted(self):
        blocks = [
            Block(kind="heading", text="8.1 站点组【客诉虚假点送达】降星", level=4),
            Block(kind="para", text="降星规则：满足以下任意条件则降星。"),
            Block(kind="heading", text="8.2 站点组【B端客诉事件】降星", level=4),
        ]

        notes = normalize_blocks(blocks)

        table = next(b for b in blocks if b.kind == "table" and b.rows)
        flat = "\n".join("\t".join(r) for r in table.rows)
        self.assertIn("虚假点送达指", flat)
        self.assertIn("客户通过拨打客服电话", flat)
        self.assertIn("客诉虚假点送达率", flat)
        self.assertIn("考核范围", flat)
        self.assertTrue(any("客诉虚假点送达定义补全" in note for note in notes))

    def test_ka_experience_adjustment_tail_table_is_repaired(self):
        blk = Block(kind="table", rows=[
            ["", "要求目标值", "麦肯必单量占比15%", "膨胀系数<br>1.2"],
            ["", "介于门槛值、目标值之间", "介于 5%到15%之间", "等比例计算膨胀系数"],
            ["", "门槛值", "5%", "0.8"],
            ["计分规则 5", "A品牌体验总得<br>肯德基体验得分值膨胀系数。封顶",
             "麦当劳完单量占比 * 麦当劳体验得分+ 必胜客完单量占比", "+ 肯德基完单量占* 麦肯必单量占"],
            ["补充说明", "亥月站点组", "含KA導企客集约", "考材包"],
            ["公示流程", "欠月的第六个工作白通过渠道经理对", "日前会公示考核结果", "可于公示次日 16点知为准。"],
        ])

        normalize_blocks([blk])

        self.assertEqual(blk.rows[0], ["项目", "内容"])
        self.assertEqual(blk.rows[1], ["计分规则4-目标值", "麦肯必单量占比：15%；膨胀系数：1.2"])
        self.assertIn("KA品牌体验总得分", blk.rows[4][1])
        self.assertIn("含KA集约站、加盟大网站，不含企客集约站", blk.rows[5][1])
        self.assertIn("次月的第六个工作日前会公示考核结果", blk.rows[6][1])

    def test_fine_schedule_supplement_table_is_repaired(self):
        blk = Block(kind="table", rows=[
            ["考核补充说明", "1.本月新建站，行判断，上月天开始考核精：",
             "考核目标值站点核目标值", "不参与本月考核"],
            ["", "6.依据站点的监控：烽火台.中，历史已发日、6月20日预估存在压力",
             "精细化排班", "7.对于考核日"],
        ])

        normalize_blocks([blk])

        self.assertEqual(blk.rows[0], ["序号", "考核补充说明"])
        self.assertEqual(len(blk.rows), 8)
        self.assertIn("站点首单日起7天内的数据不进行考核", blk.rows[1][1])
        self.assertIn("2026年6月30日数据为准", blk.rows[4][1])
        self.assertIn("端午节（6月19日、6月20日、6月21日）", blk.rows[6][1])
        self.assertIn("极小站目标值=目标值+5pp", blk.rows[7][1])

    def test_ka_supplement_common_ocr_typos_are_normalized(self):
        text, _ = normalize_text("亥月站点组，含KA導企客集约，品牌方不考材包，欠月工作白", "table")

        self.assertNotIn("亥月", text)
        self.assertNotIn("KA導", text)
        self.assertNotIn("考材", text)
        self.assertIn("考核月站点组", text)
        self.assertIn("KA及企客集约", text)
        self.assertIn("不考核门店", text)
        self.assertIn("次月工作日", text)

    def test_star_sequence_is_readable_when_used_as_a_cell_label(self):
        self.assertEqual(
            normalize_text("5星4星3星2星1星", "table")[0],
            "5星、4星、3星、2星、1星",
        )

    def test_possible_truncation_marks_block(self):
        blk = Block(kind="para", text="若单月内同一城市商虚假早会发生超过10次，并按照实际")
        normalize_blocks([blk])
        self.assertIn("possible_truncation", blk.flags)

    def test_table_noise_rows_are_removed(self):
        rows = [
            ["项目", "内容", "责任承担"],
            ["美团配送", "【保密资料，请遵守保密原则】", "MTPS-ZS-LY-V66-20250395"],
            ["1.账号绑定", "站点未完成绑定", "200元/项/次"],
        ]
        self.assertEqual(
            clean_table_noise_rows(rows),
            [
                ["项目", "内容", "责任承担"],
                ["1.账号绑定", "站点未完成绑定", "200元/项/次"],
            ],
        )

    def test_account_watermark_is_noise(self):
        self.assertTrue(is_noise_text("bm_chenjiaqiang01）|"))
        self.assertTrue(is_noise_text("chenjlaqia"))
        self.assertTrue(is_noise_text("MTPS-ZS-LY-V66-20250395"))
        self.assertFalse(is_noise_text("站点标准化率=Σ 检核项得分"))
        self.assertFalse(is_noise_text("合作商管理人员需每月检查chenjiaqiang01）要求标准站提交"))
        self.assertFalse(is_noise_text(
            "MTPS-ZS-FJ-V68-20260130及编号MTPS-ZS-FJ-V68-20260131，以下简称“主制度”"
        ))

    def test_watermark_fragment_makes_table_suspicious(self):
        rows = [
            ["检核项目", "内容", "责任承担"],
            ["强（bm_ch<br>5. 功能区配置强", "站点功能区配置要求", "200元/项/次"],
        ]
        self.assertGreaterEqual(table_suspect_score(rows), 3)

    def test_unreliable_image_caption_is_hidden(self):
        blk = Block(
            kind="image",
            text="1\n二、适用区域⋯… 1三、定义/名词解释. 2四、考核内容. 3",
            image_path="figure.png",
        )

        notes = normalize_blocks([blk])

        self.assertEqual(blk.text, "")
        self.assertIsNone(blk.rows)
        self.assertTrue(any("图片说明不可靠已隐藏" in note for note in notes))

    def test_broken_image_caption_is_hidden(self):
        blk = Block(
            kind="image",
            text=(
                "配送区域D 非KA品牌单 结算方案） 计算 非KA品牌单 体验得分\n"
                "KA星级结算金\n"
                "商服务费 结算\n\n算金额＋活动激励金）"
            ),
            image_path="figure.png",
        )

        notes = normalize_blocks([blk])

        self.assertEqual(blk.text, "")
        self.assertIsNone(blk.rows)
        self.assertTrue(any("图片说明不可靠已隐藏" in note for note in notes))

    def test_formula_image_caption_rows_are_hidden_when_broken(self):
        blk = Block(
            kind="image",
            text="",
            rows=[
                ["参考指标", "代表字母"],
                ["各KA 品牌单量", "K..Ka. Ks..Kx"],
                ["各KA 品牌体验得分", "KI分、Ka分、Ks分KN分"],
                ["各 KA品牌权重", "Q1.Q2"],
            ],
            image_path="figure.png",
        )

        notes = normalize_blocks([blk])

        self.assertEqual(blk.text, "")
        self.assertIsNone(blk.rows)
        self.assertTrue(any("图片说明不可靠已隐藏" in note for note in notes))

    def test_table_fallback_rows_are_not_hidden_when_suspicious(self):
        blk = Block(
            kind="image",
            flags=["table_low_confidence", "table_fallback"],
            rows=[
                ["方案", "", "", "内容", ""],
                ["考核规则", "以运单首次调度时的天", "等级为依据，按", "下方规则分普通场景和", "特殊场景考核并计算得分："],
                ["", "指标", "普通场景考核", "特殊场景考核", "备注"],
                ["", "负向反馈率", "天气等级为10", "导航距离>3 公里", ""],
                ["", "配送原因未完成率", "且导航距离≤3公里", "的正常天气单恶劣天气单（天气等级 20或30或", "40天气免责"],
                ["", "复合准时率（考核）", "", "40）HD 尾单&专送兜底", "40天气或导航距离>5公里免责"],
                ["", "复合超时时长", "", "单", "40天气或导航距离>5公里免责"],
            ],
        )

        normalize_blocks([blk])

        self.assertIsNotNone(blk.rows)
        self.assertEqual(blk.rows[0], ["指标", "普通场景考核", "特殊场景考核", "备注"])

    def test_compressed_toc_after_image_is_removed(self):
        blocks = [
            Block(kind="image", text="1", image_path="figure.png"),
            Block(kind="para", text="二、适用区域⋯… 1三、定义/名词解释. 2四、考核内容. 3"),
            Block(kind="heading", text="一、背景", level=2),
        ]

        notes = normalize_blocks(blocks)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].kind, "image")
        self.assertEqual(blocks[0].text, "")
        self.assertEqual(blocks[1].text, "一、背景")
        self.assertTrue(any("压缩目录残留已删除" in note for note in notes))

    def test_toc_sequence_repairs_applicable_region_number(self):
        blocks = [
            Block(kind="heading", text="目录"),
            Block(kind="para", text="一、背景.. ：1"),
            Block(kind="para", text="一、； 适用区域 1"),
            Block(kind="para", text="三、方案细则"),
            Block(kind="para", text="2"),
        ]

        notes = normalize_blocks(blocks)

        self.assertEqual([blk.text for blk in blocks], [
            "目录",
            "一、背景　1",
            "二、适用区域 1",
            "三、方案细则　2",
        ])
        self.assertTrue(any("适用区域目录编号" in note for note in notes))
        self.assertTrue(any("目录页码断行" in note for note in notes))

    def test_broken_number_clause_is_merged(self):
        blocks = [
            Block(kind="para", text="本协议内容为《制度》（编号"),
            Block(kind="para", text="MTPS-ZS-FJ-V68-20260130及编号MTPS-ZS-FJ-V68-20260131，以下简称“主制"),
            Block(kind="para", text="度”）第四条第三款。"),
        ]

        notes = normalize_blocks(blocks)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(
            blocks[0].text,
            "本协议内容为《制度》（编号 MTPS-ZS-FJ-V68-20260130及编号 MTPS-ZS-FJ-V68-20260131，以下简称“主制度”）第四条第三款。",
        )
        self.assertTrue(any("编号段落断行合并" in note for note in notes))

    def test_toc_orphan_entry_is_repaired(self):
        blocks = [
            Block(kind="heading", text="目录", level=1),
            Block(kind="para", text="一、"),
            Block(kind="para", text="二、管理规则　2"),
            Block(kind="para", text="三、问责标准　3"),
            Block(kind="heading", text="四、附则", level=2),
            Block(kind="heading", text="一、 总则", level=2),
        ]

        notes = normalize_blocks(blocks)

        self.assertEqual(blocks[1].text, "一、总则　1")
        self.assertEqual(blocks[4].kind, "para")
        self.assertEqual(blocks[4].text, "四、附则　7")
        self.assertTrue(any("目录缺失条目补全" in note for note in notes))
        self.assertTrue(any("目录标题误判修复" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
