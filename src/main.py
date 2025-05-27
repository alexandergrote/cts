import hydra
import warnings

from omegaconf import DictConfig, OmegaConf

# load package specific code
from src.util.dynamic_import import DynamicImport
from src.util.constants import EnvMode, HYDRA_CONFIG, get_hydra_output_dir
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

    # display caching warning
    if cfg['env']['cached_functions'] is None:
        cfg['env']['cached_functions'] = []

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

    y_train = output.get('y_train')
    y_test = output.get('y_test')

    if y_train is not None:
        print(y_train.value_counts(normalize=True))
        
    if y_test is not None:
        print(y_test.value_counts(normalize=True))

    # add config to kwargs
    # needed for exporting
    output['config'] = cfg

    # adding run time and memory consumption to metrics
    key_feat_sel = 'feature_selection_duration'
    key_feat_sel_mem = 'feature_selection_max_memory'
    key_n_feat_selected = 'n_features_selected'
    key_delta_confidence = 'delta_confidence_duration'
    key_delta_confidence_mem = 'delta_confidence_max_memory'

    if (output.get(key_feat_sel) is None) and (output.get(key_delta_confidence) is None):
        output['metrics'][key_feat_sel] = -1
        output['metrics'][key_feat_sel_mem] = -1

    elif (output.get(key_feat_sel) is not None) and (output.get(key_delta_confidence) is None):
        output['metrics'][key_feat_sel] = output.get(key_feat_sel)
        output['metrics'][key_feat_sel_mem] = output.get(key_feat_sel_mem)

    elif (output.get(key_feat_sel) is None) and (output.get(key_delta_confidence) is not None):
        output['metrics'][key_feat_sel] = output.get(key_delta_confidence)
        output['metrics'][key_feat_sel_mem] = output.get(key_delta_confidence_mem)
    
    else:
        raise ValueError('Invalid state: both feat_sel and delta_confidence are not None')

    
    output['metrics'][key_n_feat_selected] = output['x_test'].shape[1]
    
    if env.mode == EnvMode.PROD:
        exporter.export(output_dir=output_dir, **output)


if __name__ == '__main__':
    main()