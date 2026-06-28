import unittest

from ptt.models import Block
from ptt.qa import (doc_vote_fix, _is_clean_low_confidence_text, _is_pure_noise_text,
                    _is_safe_low_confidence_formula, _looks_like_multiline_text_formula,
                    _should_image_fallback_single)


class QaFallbackTests(unittest.TestCase):
    def test_pure_watermark_fragment_is_noise(self):
        self.assertTrue(_is_pure_noise_text("chenjiaqiang01）|"))
        self.assertTrue(_is_pure_noise_text("陈加强."))
        self.assertTrue(_is_pure_noise_text("ango1l（bm_Cn"))
        self.assertTrue(_is_pure_noise_text("# 10901）"))
        self.assertTrue(_is_pure_noise_text("an9"))
        self.assertTrue(_is_pure_noise_text("一aaaa"))
        self.assertFalse(_is_pure_noise_text("合作商标准化检查管理办法"))

    def test_long_low_confidence_text_falls_back_to_image(self):
        blk = Block(
            kind="para",
            text="攻尔统-怀维冲息信尽俁决的新建时间定12月2日，则始息维度的保护同易定12月2日",
            confidence=0.3,
            flags=["low_confidence"],
        )
        self.assertTrue(_should_image_fallback_single(blk))

    def test_plain_text_formula_does_not_fall_back_to_image(self):
        blk = Block(
            kind="para",
            text="商服务费 = 基础服务费 + 超额达标奖励 + KA星级结算金额 + KA体验膨胀费",
            confidence=0.3,
            flags=["low_confidence"],
        )
        self.assertFalse(_should_image_fallback_single(blk))
        self.assertTrue(_looks_like_multiline_text_formula(
            "商服务费 = 基础服务费 + 超额达标奖励 + KA星级结算金额\n"
            "+ KA体验膨胀费 + 服务质量奖励费 + 活动激励金"
        ))

    def test_clean_section_heading_is_allowed_despite_low_confidence(self):
        self.assertTrue(_is_clean_low_confidence_text("二、 适用区域."))
        self.assertTrue(_is_clean_low_confidence_text("二、补充方案细则…"))
        self.assertTrue(_is_clean_low_confidence_text("1.站点ID： 站点名称"))
        self.assertFalse(_is_clean_low_confidence_text("二、 适用区域 chenjiaqiang01"))

    def test_low_confidence_text_formula_is_not_blocking(self):
        self.assertTrue(_is_safe_low_confidence_formula(
            "计分规则 KA 品牌驻点骑手考核得分=W1得分*W1权重+W2得分*W2权重"
        ))

    def test_doc_vote_fix_does_not_change_letter_variables_to_zero(self):
        blocks = [
            Block(kind="para", text="0n 0n 0n"),
            Block(kind="para", text="SUM(Kn * On)"),
        ]

        doc_vote_fix(blocks)

        self.assertEqual(blocks[1].text, "SUM(Kn * On)")


if __name__ == "__main__":
    unittest.main()
