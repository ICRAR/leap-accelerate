import typer
import leap.cli.batch
import leap.cli.realtime


app = typer.Typer()
app.add_typer(leap.cli.batch.batch_app, name="batch")
app.add_typer(leap.cli.realtime.realtime_app, name="realtime")


if __name__ == "__main__":
    app()
