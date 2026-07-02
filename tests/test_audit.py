import os
import tempfile
import unittest

from ptt.audit import scan_markdown, select_sample
from ptt.coverage import _missing_numeric_anchors, _missing_seen_terms


class AuditTests(unittest.TestCase):
    def test_select_sample_wraps_around(self):
        pdfs = ["a.pdf", "b.pdf", "c.pdf"]
        self.assertEqual(select_sample(pdfs, sample_size=4, offset=2),
                         ["c.pdf", "a.pdf", "b.pdf"])

    def test_scan_markdown_finds_bad_patterns_and_table_fallbacks(self):
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

        self.assertGreaterEqual(result["issue_count"], 2)
        self.assertEqual(result["table_fallbacks"], 2)
        self.assertEqual(
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

    def test_scan_markdown_finds_salary_policy_residuals(self):
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
        self.assertIn("计分语句误识别", issue_types)
        self.assertIn("薪动力规则残留", issue_types)

    def test_scan_markdown_does_not_treat_normal_formula_as_zero_reversal(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("$$\\text{达成率}=100=100\\%$$\n")

            result = scan_markdown(path)

        issue_types = [issue["type"] for issue in result["issues"]]
        self.assertNotIn("薪动力规则残留", issue_types)

    def test_scan_markdown_allows_normal_y2_variable(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("有效在线时长≥Y2小时，标准2服务质量达标。\n")

            result = scan_markdown(path)

        self.assertEqual(result["issue_count"], 0)

    def test_scan_markdown_finds_ka_critical_missing_sections(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "#### 5.4.7 复合超时时长\n"
                    "| 订单类型 | 复合超时时长订单口径 |\n"
                    "|---|---|\n"
                    "#### 7.1 KA 品牌体验调分项\n"
                    "计分规则1：\n"
                    "#### 8.1 站点组【客诉虚假点送达】降星\n"
                    "降星规则：\n"
                )

            result = scan_markdown(path)

        critical = [i for i in result["issues"] if i["type"] == "关键内容缺失"]
        self.assertEqual(len(critical), 3)

    def test_scan_markdown_accepts_complete_ka_critical_sections(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "#### 5.4.7 复合超时时长\n"
                    "$$\\text{复合超时时长}=\\frac{A1_{\\text{KA品牌单}}+A2_{\\text{KA品牌单}}+A3_{\\text{KA品牌单}}}{W_{\\text{KA品牌单}}}$$\n"
                    "#### 7.1 KA 品牌体验调分项\n"
                    "得分范围：[0,0.2]\n"
                    "麦当劳完单量占比=麦当劳完单量/总量\n"
                    "#### 8.1 站点组【客诉虚假点送达】降星\n"
                    "虚假点送达指骑手未将餐品/货品按照订单要求送达指定位置虚假点击送达的行为。\n"
                    "客诉虚假点送达率=命中客诉虚假点送达单量/完成单量\n"
                )

            result = scan_markdown(path)

        issue_types = [issue["type"] for issue in result["issues"]]
        self.assertNotIn("关键内容缺失", issue_types)

    def test_scan_markdown_finds_table_readability_regressions(self):
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
        self.assertIn("表格结构可读性问题", issue_types)

    def test_coverage_allows_normalized_weather_and_unit_noise(self):
        self.assertEqual(
            _missing_seen_terms(
                "天气等级20恶劣天气单",
                "天气等级为20或30或40恶劣天气单",
                ["天气等级20", "恶劣天气单"],
            ),
            [],
        )
        self.assertEqual(
            _missing_numeric_anchors(
                "40天气或240天气免责，得分 120S",
                "40天气免责，得分：120",
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
