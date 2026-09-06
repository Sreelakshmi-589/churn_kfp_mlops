from kfp import dsl
from kfp import local

from pipeline.components.load_data import load_data
from pipeline.components.validate import validate_data
from pipeline.components.preprocess import preprocess_data
from pipeline.components.train import train_model
from pipeline.components.evaluate import evaluate_model
from pipeline.components.quality_gate import quality_gate


local.init(
    runner=local.SubprocessRunner()
)


@dsl.pipeline(
    name="churn-mlops-pipeline"
)
def churn_pipeline():

    load_task = load_data(
        input_path="data/churn.csv"
    )

    validate_task = validate_data(
        input_data=load_task.outputs["output_data"]
    )

    preprocess_task = preprocess_data(
        input_data=load_task.outputs["output_data"]
    )
    
    train_task = train_model(
        train_data=preprocess_task.outputs["train_data"]
    )
    evaluate_task = evaluate_model(
        model=train_task.outputs["model"],
        test_data=preprocess_task.outputs["test_data"],
        threshold=0.60,
    )


    validate_task.after(load_task)
    preprocess_task.after(validate_task)
    train_task.after(preprocess_task)
    evaluate_task.after(train_task)
    

if __name__ == "__main__":
    churn_pipeline()
