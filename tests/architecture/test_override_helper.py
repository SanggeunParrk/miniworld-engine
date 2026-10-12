"""``override_setting`` converts strings by field type and restores the previous settings after the test."""

from settings_helpers import override_setting

from miniworld_engine import settings


def test_bool_from_strings(monkeypatch):
    override_setting(monkeypatch, "bias_only_dit_inf3_bf16", "0")
    assert settings.current().bias_only_dit_inf3_bf16 is False
    override_setting(monkeypatch, "bias_only_dit_inf3_bf16", "1")
    assert settings.current().bias_only_dit_inf3_bf16 is True


def test_optional_int_and_none_restores_default(monkeypatch):
    override_setting(monkeypatch, "bias_only_dit_pv_vb", "256")
    assert settings.current().bias_only_dit_pv_vb == 256
    override_setting(monkeypatch, "bias_only_dit_pv_vb", None)
    assert settings.current().bias_only_dit_pv_vb is None


def test_tri_state(monkeypatch):
    override_setting(monkeypatch, "opm_train", "0")
    assert settings.current().opm_train is False


def test_the_previous_settings_come_back():
    # the monkeypatch of the tests above restored the module state at their teardown
    current = settings.current()
    assert current.bias_only_dit_inf3_bf16 is True
    assert current.bias_only_dit_pv_vb is None
    assert current.opm_train is None
