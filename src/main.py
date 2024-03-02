import hydra
from omegaconf import DictConfig, OmegaConf

# load package specific code
from src.util.dynamic_import import DynamicImport
from src.util.constants import Directory, File
from src.util.logging import console


@hydra.main(
    config_path=str(Directory.CONFIG),
    config_name=File.CONFIG.stem,
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:

    console.rule("Executing experiment")

    cfg = OmegaConf.to_container(cfg)

    data_loader = DynamicImport.import_class_from_dict(
        dictionary=cfg['fetch_data']
    )

    console.log("Fetching data")
    output = data_loader.execute()

    preprocessor = DynamicImport.import_class_from_dict(
        dictionary=cfg['preprocess']
    )

    console.log("Starting Preprocessing")
    output = preprocessor.execute(**output)
    console.log("Finished Preprocessing")

    model = DynamicImport.import_class_from_dict(
        dictionary=cfg['model']
    )

    console.log("Model Training")
    output = model.fit(**output)
    console.log("Model Interference")
    output = model.predict(**output)
    console.log("Model Interference with Probabilities")
    output = model.predict_proba(**output)

    evaluator = DynamicImport.import_class_from_dict(
        dictionary=cfg['evaluation']
    )

    console.log("Evaluating")
    output = evaluator.evaluate(**output)
    console.log(output.get('metrics'))

    print(output.get('y_train').value_counts(normalize=True))
    print(output.get('y_test').value_counts(normalize=True))

    exporter = DynamicImport.import_class_from_dict(
        dictionary=cfg['export']
    )

    # add config to kwargs
    # needed for exporting
    output['config'] = cfg

    exporter.export(**output)


if __name__ == '__main__':
    main()