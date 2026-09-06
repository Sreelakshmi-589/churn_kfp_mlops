from kfp import dsl


@dsl.component(
    base_image="python:3.11"
)
def quality_gate(
    f1: float,
    threshold: float,
) -> str:

    print("===================================")
    print("MODEL QUALITY GATE")
    print("===================================")
    print(f"F1 Score : {f1:.4f}")
    print(f"Threshold: {threshold:.4f}")

    if f1 < threshold:
        raise ValueError(
            f"MODEL REJECTED: "
            f"F1 score {f1:.4f} is below "
            f"threshold {threshold:.4f}"
        )

    print(
        f"MODEL APPROVED: "
        f"F1 score {f1:.4f} >= "
        f"threshold {threshold:.4f}"
    )

    print("===================================")

    return "APPROVED"
