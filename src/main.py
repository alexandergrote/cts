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


if __name__ == '__main__':
    main()