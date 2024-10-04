import typer
from src.util.config_import import YamlConfigLoader
from src.util.datasets import DatasetSchema
from src.util.dynamic_import import DynamicImport

app = typer.Typer()

@app.command()
def eda(dataset: str = 'synthetic', overrides: str = ''):
    """
    Perform exploratory data analysis.

    Args:
    dataset (str): dataset key in yaml configs.
    overrides (str): The configuration overrides. Defaults to empty string.
    """

    final_overrides = [f"fetch_data={dataset}"]
    
    if overrides != '':
        final_overrides += overrides.split(' ')


    config = YamlConfigLoader.read_yaml(key="fetch_data", overrides=final_overrides)
    data_loader = DynamicImport.import_class_from_dict(config)
    data = data_loader.execute()['data']

    # number of observations, equivalent to number of unique ids
    typer.echo("Number of unique observations per column:")
    typer.echo(data[data[DatasetSchema.id_column].notnull()].nunique())

    typer.echo("Unique events:")
    typer.echo(data[DatasetSchema.event_column].nunique())

    # top 10 unique events by frequency
    typer.echo("Top 10 unique events by frequency:")
    typer.echo(data[DatasetSchema.event_column].value_counts().head(10))

    typer.echo("Event counts:")
    typer.echo(data[DatasetSchema.event_column].value_counts(normalize=False).mean())

    typer.echo("Sequence length:")
    typer.echo(data.groupby(DatasetSchema.id_column).size().mean())

    typer.echo("Min sequence length:")
    typer.echo(data.groupby(DatasetSchema.id_column).size().min())

    typer.echo("Max sequence length:")
    typer.echo(data.groupby(DatasetSchema.id_column).size().max())

    typer.echo("Class distribution:")
    typer.echo(data[DatasetSchema.class_column].mean())

if __name__ == "__main__":

    # python src\fetch_data\util\eda.py --dataset malware --overrides "fetch_data.params.resampling_fraction=null"

    app()

