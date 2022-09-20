from datasets import load_metric
from bert_utils import prepare_validation_features
import subprocess
from typing import NamedTuple
import subprocess

class BenchmarkOutput(NamedTuple):
    hint: str
    device: str
    shape: str
    latency: float
    throughput: float

def benchmark_model(model_path, device, seconds, hint, input_shape):
    benchmark_command = f"benchmark_app -m {model_path} -d {device} -t {seconds} -hint {hint} -shape {input_shape} --report_type no_counters --report_folder ." 
    # print(benchmark_command)
    try:
        subprocess.run(
            benchmark_command.split(" "),
            capture_output=True,
            universal_newlines=True
        )
    except:
        raise

    with open("benchmark_report.csv") as f:
        benchmark_output = f.read()
    
    latency = next((line.split(";")[-1] for line in benchmark_output.splitlines() if line.startswith("latency")), None)
    throughput = next((line.split(";")[-1] for line in benchmark_output.splitlines() if line.startswith("throughput")), None)

    return BenchmarkOutput(hint=hint, device=device, shape=input_shape, latency=latency, throughput=throughput)

def benchmark_model_parse(model_path, device, seconds, hint, input_shape):
    benchmark_command = f"benchmark_app -m {model_path} -d {device} -t {seconds} -hint {hint} -shape {input_shape}" 
    # print(benchmark_command)
    try:
        benchmark_output = subprocess.run(
            benchmark_command.split(" "),
            capture_output=True,
            universal_newlines=True
        )
    except:
        # print(f"Failed: {model_path}")
        # print(benchmark_output)
        return -1
    
    try:
        benchmark_result = [
            line
            for line in benchmark_output.stdout.splitlines()
            if not (line.startswith(r"[") or line == "")
        ]
        # print(benchmark_result)
    except IndexError:
        # print(f"Benchmark failed. Full output: {benchmark_output.stdout.splitlines()}")
        return -1
    try:
        latency = (
            [line for line in benchmark_result if "Median" in line][0]
            .split(":")[1]
            .strip()
        )
        # print("latency", latency)
        latency = latency.split(" ")[0]
    except:
        # print(f"Failed: {model_path}")
        # print(benchmark_output.stdout)
        # print(benchmark_output.stderr)
        return -1
    return latency



def compute_metrics(pred, examples) -> dict:
    import transformers
    metric = load_metric("squad")
    validation_features = examples.map(prepare_validation_features, batched=True, remove_columns=examples.column_names)

    # The Trainer hides the columns that are not used by the model (here example_id and offset_mapping which we will need for our post-processing), so we set them back
    validation_features.set_format(
        type=validation_features.format["type"],
        columns=list(validation_features.features.keys()),
    )

    # To get the final predictions we can apply our post-processing function to our raw predictions
    final_predictions = postprocess_qa_predictions(examples, validation_features, pred.predictions)

    # We just need to format predictions and labels a bit as metric expects a list of dictionaries and not one big dictionary
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]

    # Hide again the columns that are not used by the model
    validation_features.set_format(
        type=validation_features.format["type"],
        columns=["attention_mask", "end_positions", "input_ids", "start_positions"],
    )
    metrics = metric.compute(predictions=formatted_predictions, references=references)

    return metrics