from __future__ import annotations

import typer

from data_for_tsfms.cli.evaluate_fev import main as evaluate_fev_main
from data_for_tsfms.cli.evaluate import main as evaluate_main
from data_for_tsfms.cli.prepare_data import main as prepare_main
from data_for_tsfms.cli.train import main as train_main

app = typer.Typer(add_completion=False, help="TSFMS CLI")

app.command(name="prepare")(prepare_main)
app.command(name="train")(train_main)
app.command(name="evaluate")(evaluate_main)
app.command(name="evaluate-fev")(evaluate_fev_main)


if __name__ == "__main__":
    app()
