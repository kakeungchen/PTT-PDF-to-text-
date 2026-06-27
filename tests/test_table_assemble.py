import unittest

from ptt.assemble import (TableRegion, _merge_table_fallback_fragments,
                          build_table_block, detect_page_bands,
                          strip_header_footer)
from ptt.models import Block, Line


class TableAssembleTests(unittest.TestCase):
    def test_suspect_table_falls_back_to_image_when_amount_enters_non_last_column(self):
        region = TableRegion(
            y0=0, y1=90,
            hy=[0, 30, 60, 90],
            vx=[0, 100, 300, 430],
            x0=0, x1=430,
        )
        lines = [
            Line("项目", 0.99, 10, 8, 60, 22),
            Line("内容", 0.99, 120, 8, 170, 22),
            Line("责任承担", 0.99, 320, 8, 390, 22),
            Line("4.站点经营", 0.92, 10, 38, 85, 52),
            Line("1、严禁存放非美团装备。200元/项/次，整改不达标", 0.75, 120, 38, 290, 52),
            Line("达标需承担双倍违约金。", 0.9, 320, 38, 410, 52),
        ]
        block, used = build_table_block(region, lines, page=0)
        self.assertIsNotNone(block)
        self.assertEqual(block.kind, "image")
        self.assertIn("table_low_confidence", block.flags)
        self.assertIn("table_fallback", block.flags)
        self.assertEqual(len(used), len(lines))

    def test_table_fallback_absorbs_adjacent_table_fragments(self):
        blocks = [
            Block(kind="image", bbox=(0, 100, 500, 200), page=0,
                  flags=["table_low_confidence", "table_fallback"]),
            Block(kind="para", text="自主餐箱消毒管理看板 场景 3 【60%，70%） 1500元/站/月",
                  bbox=(20, 220, 480, 250), page=0),
            Block(kind="heading", text="1. 包括但不限于非实景拍摄、无",
                  bbox=(20, 265, 480, 290), page=0),
            Block(kind="table", rows=[["场景 4", "【50%，60%）", "2000元/站/月"]],
                  bbox=(20, 305, 480, 340), page=0),
            Block(kind="heading", text="1. 标准站系统（烽火台-组织区域--站点基础建设）",
                  bbox=(20, 345, 480, 355), page=0),
            Block(kind="image", text="整改不达标需承担", rows=[["站", "建设", "双倍违约金。"]],
                  bbox=(0, 358, 500, 370), page=0),
            Block(kind="image", bbox=(0, 360, 500, 430), page=0,
                  flags=["table_low_confidence", "table_fallback"]),
            Block(kind="heading", text="4.2 区域商管控规则",
                  bbox=(20, 520, 480, 550), page=0),
        ]

        merged = _merge_table_fallback_fragments(blocks, page_w=500)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].kind, "image")
        self.assertIn("merged_table_fallback", merged[0].flags)
        self.assertEqual(merged[0].bbox, (0, 100, 500, 430))
        self.assertEqual(merged[1].text, "4.2 区域商管控规则")

    def test_table_fallback_keeps_case_explanation_as_text(self):
        blocks = [
            Block(kind="image", bbox=(0, 100, 500, 200), page=0,
                  flags=["table_low_confidence", "table_fallback"]),
            Block(kind="para", text="案例说明：例1：区域商当月服装标准化率为89%，核算违约金额1000元。",
                  bbox=(20, 230, 480, 260), page=0),
        ]

        merged = _merge_table_fallback_fragments(blocks, page_w=500)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].kind, "image")
        self.assertEqual(merged[1].kind, "para")

    def test_nearby_table_fragment_becomes_small_fallback_image(self):
        blocks = [
            Block(kind="image", bbox=(0, 100, 500, 200), page=0,
                  flags=["table_low_confidence", "table_fallback"]),
            Block(kind="heading", text="2. 烟感状态要求全部在线（看闪灯即 双倍违约金。",
                  bbox=(20, 1700, 480, 1740), page=0),
        ]

        merged = _merge_table_fallback_fragments(blocks, page_w=500)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1].kind, "image")
        self.assertIn("fragment_table_fallback", merged[1].flags)

    def test_repeated_account_watermark_is_removed_across_page(self):
        lines = [
            Line("bm_chenjiaqiang01）|", 0.35, 10, 100, 160, 120),
            Line("chenjiaqiang01）", 0.35, 220, 580, 360, 600),
            Line("chenjiaqiang01）", 0.35, 420, 980, 560, 1000),
            Line("chenjiaqiang01）", 0.35, 40, 1500, 180, 1520),
            Line("chenjiaqiang01）", 0.35, 340, 1980, 480, 2000),
            Line("正文内容应该保留", 0.98, 20, 2200, 240, 2220),
        ]
        bands, notes = detect_page_bands(lines, page_h=2600)
        kept, removed = strip_header_footer(lines, bands)

        self.assertGreaterEqual(removed, 5)
        self.assertEqual([l.text for l in kept], ["正文内容应该保留"])
        self.assertTrue(any("账号水印" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
