from kfp import dsl
from kfp.dsl import Output, Dataset


@dsl.component(
    base_image="python:3.11"
)
def load_data(
    input_path: str,
    output_data: Output[Dataset],
):
    import shutil
    from pathlib import Path

    source = Path(input_path)
    destination = Path(output_data.path)

    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        source,
        destination
    )

    print("===================================")
    print("DATA LOADED SUCCESSFULLY")
    print("===================================")
    print(f"Source: {source}")
    print(f"Output artifact: {destination}")
    print("===================================")
