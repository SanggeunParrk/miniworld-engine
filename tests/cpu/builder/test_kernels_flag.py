"""`build --kernels {auto,native,portable}`: the only spelling; the former `--backend` alias is gone."""
import pytest

from miniworld_engine import cli


def _choice(*extra):
    args = cli.build_parser().parse_args(["build", "all", *extra])
    return cli._kernels_choice(args)


def test_the_default_is_auto():
    assert _choice() == "auto"


@pytest.mark.parametrize("value", ["auto", "native", "portable"])
def test_kernels_takes_the_three_names(value):
    assert _choice("--kernels", value) == value



def test_triton_is_not_a_choice_of_kernels():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["build", "all", "--kernels", "triton"])


@pytest.mark.parametrize("value", ["auto", "native", "triton"])
def test_the_old_backend_spelling_is_gone(value, capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["build", "all", "--backend", value])
