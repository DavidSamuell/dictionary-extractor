"""Tests for dictionary language config and MDF marker assignment."""

from __future__ import annotations

from dictextractor.utils.dictionary_languages import (
    build_config_from_folder,
    config_to_yaml_dict,
    markers_for_config,
    mdf_marker_for_target,
)


def test_mdf_marker_english() -> None:
    assert mdf_marker_for_target("en") == "ge"


def test_mdf_marker_national_languages() -> None:
    assert mdf_marker_for_target("ru", non_english_index=0) == "gn"
    assert mdf_marker_for_target("fr", non_english_index=0) == "gn"
    assert mdf_marker_for_target("zh", non_english_index=0) == "gn"


def test_mdf_marker_second_non_english() -> None:
    assert mdf_marker_for_target("tr", non_english_index=1) == "gr"


def test_yaml_has_no_mdf_markers() -> None:
    config = build_config_from_folder("Chukchi-Russian", metadata_rows=[])
    yaml_dict = config_to_yaml_dict(config)
    assert "mdf_marker" not in str(yaml_dict)
    assert "mdf_lexeme" not in str(yaml_dict)


def test_markers_for_config_fallback() -> None:
    config = build_config_from_folder("Chukchi-Russian", metadata_rows=[])
    assert markers_for_config(config) == {"ru": "gn"}


def test_chukchi_russian_targets() -> None:
    config = build_config_from_folder("Chukchi-Russian", metadata_rows=[])
    assert len(config.targets) == 1
    assert config.targets[0].code == "ru"


def test_na_english_chinese_trilingual() -> None:
    config = build_config_from_folder("Na-English-Chinese", metadata_rows=[])
    codes = [t.code for t in config.targets]
    assert codes == ["en", "zh"]
    assert markers_for_config(config) == {"en": "ge", "zh": "gn"}


def test_circassian_english_turkish_columns() -> None:
    config = build_config_from_folder("Circassian-English-Turkish", metadata_rows=[])
    yaml_dict = config_to_yaml_dict(config)
    assert yaml_dict["source"]["column_id"] == "center"
    assert yaml_dict["targets"][0]["column_id"] == "left"


def test_vernacular_syriac_kurdish_english() -> None:
    config = build_config_from_folder(
        "Vernacular Syriac-Kurdish_Turkish-English",
        metadata_rows=[],
    )
    assert markers_for_config(config) == {"ku_tr": "gn", "en": "ge"}
