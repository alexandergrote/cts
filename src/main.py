import hydra
import warnings

from omegaconf import DictConfig, OmegaConf

# load package specific code
from src.util.dynamic_import import DynamicImport
from src.util.constants import Directory, File, replace_placeholder_in_dict
from src.util.custom_logging import console

# Ignore all runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(
    config_path=str(Directory.CONFIG),
    config_name=File.CONFIG.stem,
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:

    console.rule("Executing experiment")

    cfg = OmegaConf.to_container(cfg)

    for _, value in cfg['constants'].items():

        placeholder = value['placeholder']
        replacement = value['value']

        cfg = replace_placeholder_in_dict(
            dictionary=cfg,
            placeholder=placeholder,
            replacement=replacement
        )

    # loading all components
    data_loader = DynamicImport.import_class_from_dict(
        dictionary=cfg['fetch_data']
    )

    preprocessor = DynamicImport.import_class_from_dict(
        dictionary=cfg['preprocess']
    )

    train_test_split = DynamicImport.import_class_from_dict(
        dictionary=cfg['train_test_split']
    )

    model = DynamicImport.import_class_from_dict(
        dictionary=cfg['model']
    )

    evaluator = DynamicImport.import_class_from_dict(
        dictionary=cfg['evaluation']
    )

    exporter = DynamicImport.import_class_from_dict(
        dictionary=cfg['export']
    )

    # execute all components

    console.log("Fetching data")
    output = data_loader.execute()

    console.log("Starting Preprocessing")
    output = preprocessor.execute(**output)
    console.log("Finished Preprocessing")

    console.log("Splitting data")
    output = train_test_split.execute(**output)

    console.log("Model Training")
    output = model.fit(**output)
    console.log("Model Interference")
    output = model.predict(**output)
    console.log("Model Interference with Probabilities")
    output = model.predict_proba(**output)

    console.log("Evaluating")
    output = evaluator.evaluate(**output)
    console.log(output.get('metrics'))

    print(output.get('y_train').value_counts(normalize=True))
    print(output.get('y_test').value_counts(normalize=True))

    # add config to kwargs
    # needed for exporting
    output['config'] = cfg

    exporter.export(**output)


if __name__ == '__main__':
    main()