import typer
from madgan.train import train
from madgan.preprocess import swat, convert_csv
from madgan.detect import detect

_madgan_cli = typer.Typer(name="MAD-GAN CLI", pretty_exceptions_enable=False)
_madgan_cli.command(name="train")(train)
_madgan_cli.command(name="swat")(swat)
_madgan_cli.command(name="detect")(detect)
_madgan_cli.command(name="convert")(convert_csv)


@_madgan_cli.callback()
def main() -> None:
    """MAD-GAN Command Line Interface."""


if __name__ == "__main__":
    _madgan_cli()
