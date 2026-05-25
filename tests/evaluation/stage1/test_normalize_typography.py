"""Tests for typography normalization v1."""

from dictextractor.evaluation.stage1.normalize_typography import (
    TYPOGRAPHY_SPEC_VERSION,
    is_junk_ocr_line,
    normalize_line,
)


def test_spec_version() -> None:
    assert TYPOGRAPHY_SPEC_VERSION == "v1"


def test_markdown_bold_italic() -> None:
    assert normalize_line("**head** and *tail*") == "<b>head</b> and <i>tail</i>"


def test_html_mapping() -> None:
    assert normalize_line("<strong>x</strong> <em>y</em>") == "<b>x</b> <i>y</i>"


def test_idempotent_on_dictionary_tags() -> None:
    line = "<b>ээттик</b> 1) на́рта"
    assert normalize_line(line) == line


def test_strip_unknown_html() -> None:
    assert normalize_line("<table><tr><td>cell</td></tr></table>") == "cell"


def test_latex_inline() -> None:
    out = normalize_line(r"\(\lambda\)")
    assert "lambda" in out
    assert r"\(" not in out


def test_strip_mineru_xi_template_run() -> None:
    noisy = "weak: x<i>{1}</i> x<i>{2}</i> x<i>{3}</i> tail"
    assert normalize_line(noisy) == "weak: tail"


def test_strip_latex_text_and_overline() -> None:
    assert normalize_line(r"\text{E} parallel") == "E parallel"
    assert normalize_line(r"\overline{卍}") == "卍"


def test_strip_swastika_runs() -> None:
    assert normalize_line("卍卍卍卍卍卍").strip() == ""
    assert normalize_line("ab卍卍") == "ab卍卍"


def test_no_false_italic_from_xi_garbage() -> None:
    line = "x<i>{1}</i> x<i>{2}</i> hello"
    assert "<i>" not in normalize_line(line)
    assert "hello" in normalize_line(line)


def test_html_entity() -> None:
    assert normalize_line("tsh&#x27;khâ") == "tsh'khâ"


def test_latex_mathbf_to_bold_tag() -> None:
    assert normalize_line(r"\mathbf{no}") == "<b>no</b>"
    assert normalize_line(r"\boldsymbol{-}") == "<b>-</b>"
    out = normalize_line(r"$\mathbf{n o} \boldsymbol{-}$")
    assert "<b>n o</b>" in out
    assert "<b>-</b>" in out


def test_latex_mathbf_unwrap() -> None:
    out = normalize_line(r"\mathbf{omel}\varnothing_{3} salt")
    assert out == "<b>omel</b>3 salt"
    assert r"\mathbf" not in out


def test_latex_textit_to_italic_tag() -> None:
    assert normalize_line(r"\textit{gloss}") == "<i>gloss</i>"
    assert normalize_line(r"\mathit{ipa}") == "<i>ipa</i>"


def test_mathpix_line_preserves_latex_bold() -> None:
    raw = (
        "la forme $\\mathbf{n o} \\boldsymbol{-}=\\lfloor \\rfloor$ est jugée plus correcte"
    )
    out = normalize_line(raw)
    assert "<b>n o</b>" in out
    assert "<b>-</b>" in out


def test_preserve_pos_marker() -> None:
    assert "(<N>)" in normalize_line("north (<N>)")


def test_junk_digit_line_dropped() -> None:
    assert normalize_line("12345678901234567890" * 5) == ""


def test_junk_greek_spam_dropped() -> None:
    spam = "Υ" * 300
    assert normalize_line(spam) == ""


def test_is_junk_digit() -> None:
    assert is_junk_ocr_line("9" * 100)


def test_junk_replacement_char_dropped() -> None:
    assert normalize_line("hello\ufffdworld") == ""


def test_bare_latex_subscript_unwrapped() -> None:
    assert normalize_line("kí t_{aim} ili") == "kí taim ili"


def test_braced_gloss_unwrapped() -> None:
    assert normalize_line("v.a. {heesábee shógha}") == "v.a. heesábee shógha"


def test_junk_hebrew_repeat_after_combining_stripped() -> None:
    line = "to fade, to cease bloom- vn. אַרְּ" + "רָּ" * 14
    assert normalize_line(line) == ""


def test_junk_cjk_symbol_spam_dropped() -> None:
    spam = "卄 千 十 卂 卶 卉 卋 协 卆 午 却 危 単 卧 卩 升 卪 卭 卒 卜 印 半 华 卅 单 卬 卐 博 卵 卑 卮 卌 卝 卞 卛 卙 卦 卯 區 " * 3
    assert normalize_line(spam) == ""


def test_preserve_bilingual_chinese_english_line() -> None:
    line = "word <b>你好</b> definition with enough English text here"
    assert normalize_line(line) != ""


def test_strip_hathitrust_boilerplate_in_line() -> None:
    from dictextractor.ocr.adapters.markdown_to_flat import is_boilerplate_line

    assert is_boilerplate_line(
        "Generated at University of Melbourne through HathiTrust on 2026-05-18"
    )


def test_normalize_cuneiform_spacing() -> None:
    from dictextractor.evaluation.stage1.normalize_typography import (
        normalize_cuneiform_spacing,
    )

    spaced = "¶ \U00012000 \U00012047 \U00012051 \U00012054 \U000120bc agarin."
    assert normalize_cuneiform_spacing(spaced) == "¶ \U00012000\U00012047\U00012051\U00012054\U000120bc agarin."
    assert normalize_line(spaced) == normalize_line(
        "¶ \U00012000\U00012047\U00012051\U00012054\U000120bc agarin."
    )


def test_normalize_curly_apostrophe() -> None:
    assert normalize_line("nae\u2019er") == "nae'er"


def test_normalize_unicode_dash_in_header() -> None:
    out = normalize_line("FUN \u2013 96 \u2013 page")
    assert "—" in out


def test_normalize_unicode_dash_in_compound() -> None:
    assert normalize_line("one\u2013self") == "one-self"


def test_strip_latex_mathrm_and_bare_dollar() -> None:
    out = normalize_line(r"$\mathrm{k}^{\mathrm{h}}$")
    assert "k" in out
    assert r"\mathrm" not in out


def test_strip_latex_drop_commands() -> None:
    out = normalize_line(r"no \dashv=\dot{i}")
    assert r"\dashv" not in out


def test_markdown_table_and_empty_header() -> None:
    from dictextractor.ocr.adapters.markdown_to_flat import markdown_text_to_flat_lines

    raw = "| :--- | :--- |\n| head | gloss |\n###\n"
    lines = markdown_text_to_flat_lines(raw)
    assert "head" in lines[0]
    assert "gloss" in lines[1]
    assert all(ln != "###" for ln in lines)


def test_strip_caret_text_artifact() -> None:
    out = normalize_line("avto ^text и́на")
    assert "^text" not in out and "_text" not in out
    assert "avto" in out and "и́на" in out
    assert "_text" not in normalize_line("word _text tail")
    latex = r"рот ${ }^{\text {~ }}$-са"
    assert "^text" not in normalize_line(latex).lower()


def test_strip_latex_wedge_artifact() -> None:
    assert "wedge" not in normalize_line(r"ca \wedge nx").lower()
    assert normalize_line("cawedge-nx") == "ca-nx"
    assert "wedge" not in normalize_line(r"<b>c a</b> wedge-<b>n</b>").lower()
