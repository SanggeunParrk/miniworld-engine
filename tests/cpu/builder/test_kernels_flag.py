"""`build --kernels {auto,native,portable}`: the user-facing spelling; `--backend {auto,native,triton}` is its hidden, deprecated alias."""
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


@pytest.mark.parametrize(("old", "new"), [("auto", "auto"), ("native", "native"), ("triton", "portable")])
def test_the_old_backend_spelling_still_parses(old, new):
    assert _choice("--backend", old) == new


def test_triton_is_not_a_public_choice_of_kernels():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["build", "all", "--kernels", "triton"])
