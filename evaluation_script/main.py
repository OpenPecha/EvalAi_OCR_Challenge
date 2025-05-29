import pandas as pd
from evaluate import load
import pyewts
import json
import sys


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    output = {}

    # initialize wylie and cer scorer
    converter = pyewts.pyewts()
    cer_scorer = load("cer")

    annotations = load_json_file(test_annotation_file)
    predictions = load_json_file(user_submission_file)

    # Validate: Check file sizes are same
    if len(annotations) != len(predictions):
        print(f"[ERROR] Mismatch in number of entries: annotations({len(annotations)}) != predictions({len(predictions)})")
        sys.exit(1)
    
    # Validate: Ensure all filenames match
    annotation_filenames = {entry["filename"] for entry in annotations}
    prediction_filenames = {entry["filename"] for entry in predictions}

    if annotation_filenames != prediction_filenames:
        missing_in_pred = annotation_filenames - prediction_filenames
        missing_in_annot = prediction_filenames - annotation_filenames

        if missing_in_pred:
            print(f"[ERROR] Your submission is missing predictions for these files:\n{sorted(missing_in_pred)}")
        if missing_in_annot:
            print(f"[ERROR] Your submission contains extra predictions for files not in ground truth:\n{sorted(missing_in_annot)}")
        sys.exit(1)

    # Index predictions by filename for quick lookup
    pred_dict = {entry["filename"]: entry["prediction"] for entry in predictions}

    cer_scores = []
    missed_files = 0

    for entry in annotations:
        filename = entry["filename"]
        label = entry["label"]
        prediction = pred_dict[filename]

        # Convert both label and prediction to Wylie transliteration
        label_wylie = converter.toWylie(label.strip())
        prediction_wylie = converter.toWylie(prediction.strip())

        try:
            cer_score = cer_scorer.compute(predictions=[prediction_wylie], references=[label_wylie])
            cer_scores.append(cer_score)
        except Exception as e:
            print(f"[ERROR] Failed on {filename}: {e}")
            continue

    if len(cer_scores) == 0:
        mean_cer = 1.0  # Set max CER if nothing evaluated
    else:
        mean_cer = sum(cer_scores) / len(cer_scores)

    print(f"Evaluated {len(cer_scores)} samples.")
    print(f"Mean CER: {mean_cer:.4f}")

    # Build EvalAI-compatible output
    output = {}

    metrics = {
        "CER": round(mean_cer, 4)
    }

    if phase_codename == "dev":
        output["result"] = [{"train_split": metrics}]
        output["submission_result"] = metrics
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        output["result"] = [
            {"train_split": metrics},
            {"test_split": metrics}
        ]
        # output["submission_result"] = output["result"][0]
        output["submission_result"] = output["result"][1]["test_split"]
        # i have to take out submissio result as metric cer only 
        print("Completed evaluation for Test Phase")

    return output

    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """

