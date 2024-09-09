import hydra
import warnings

from omegaconf import DictConfig, OmegaConf

# load package specific code
from src.util.dynamic_import import DynamicImport
from src.util.constants import replace_placeholder_in_dict, EnvMode, HYDRA_CONFIG, get_hydra_output_dir
from src.util.custom_logging import console
from src.util.check_experiment import experiment_exists
from src.util.environment import PydanticEnvironment

# Ignore all runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(
    **HYDRA_CONFIG,
)
def main(cfg: DictConfig) -> None:

    console.rule("Executing experiment")

    cfg = OmegaConf.to_container(cfg)

    # environment variables
    PydanticEnvironment.set_environment_variables(cfg['env'])
    env = PydanticEnvironment.create_from_environment()

    # check if experiment already exists
    experiment_name = cfg['export']['params']['experiment_name']
    random_seed = cfg['train_test_split']['params']['random_state']

    if env.mode == EnvMode.PROD:

        if experiment_exists(experiment_name=experiment_name, random_seed=random_seed):
            console.log("Experiment already exists")
            return
    

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

    # get folder for run
    output_dir = get_hydra_output_dir()

    # execute all components

    console.log("Fetching data")
    output = data_loader.execute()

    console.log("Splitting data")
    output = train_test_split.execute(**output)

    console.log("Starting Preprocessing")
    output = preprocessor.execute(**output, case_name=experiment_name)
    
    console.log("Model Training & Inference")
    model.fit(output_dir=output_dir, **output)
    output = model.predict(**output)
    output = model.predict_proba(**output)

    console.log("Evaluating")
    output = evaluator.evaluate(**output)
    console.log(output.get('metrics'))

    print(output.get('y_train').value_counts(normalize=True))
    print(output.get('y_test').value_counts(normalize=True))

    # add config to kwargs
    # needed for exporting
    output['config'] = cfg

    exporter.export(output_dir=output_dir, **output)


if __name__ == '__main__':
    main()