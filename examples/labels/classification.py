# pylint: disable=missing-module-docstring, missing-function-docstring, invalid-name
import os
import json
from tqdm import tqdm
from dotenv import load_dotenv

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from prompts.task_label import TaskLabelPrompt
from ai_eng.service.openai_service import OpenAIService

load_dotenv()

# Resolve the path to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Define clients
# -----------------------------------------------------------------------------

groq_service = OpenAIService(provider="groq")
ollama_service = OpenAIService(provider="ollama")

# -----------------------------------------------------------------------------
# Define prompts
# -----------------------------------------------------------------------------

task_label_prompt = TaskLabelPrompt()

# -----------------------------------------------------------------------------
# Define functions
# -----------------------------------------------------------------------------

def get_response_from_groq(user_message: str) -> str:
    response = groq_service.create_chat_completion(
        model="gpt-4o-mini",
        messages=task_label_prompt.compile_messages(user_message)
    )
    return response.choices[0].message.content

def get_response_from_ollama(user_message: str) -> str:
    response = ollama_service.create_chat_completion(
        model="llama3.1:8b",
        messages=task_label_prompt.compile_messages(user_message)
    )
    return response.choices[0].message.content

def evaluate_predictions(tasks: list[dict]) -> dict:
    """Evaluate model predictions using various metrics."""

    # Extract true labels and predictions
    y_true = [task['category'] for task in tasks]
    y_pred = [task['prediction'] for task in tasks]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['work', 'home', 'errands', 'general'],
                yticklabels=['work', 'home', 'errands', 'general'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(script_dir, "confusion_matrix.png"))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
    }

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    json_path = os.path.join(script_dir, "tasks.json")
    output_path = os.path.join(script_dir, "predictions.json")

    # Load tasks from JSON file
    with open(json_path, "r", encoding="utf-8") as file:
        tasks = json.load(file)

    # Classify tasks
    for task in tqdm(tasks, desc="Classifying tasks"):
        label = get_response_from_ollama(task["task"])
        task["prediction"] = label

    # Save the results to a JSON file
    print(f"Saving results to {output_path}")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(tasks, file, indent=4)

    # Evaluate the predictions
    metrics = evaluate_predictions(tasks)
    print(f"Saving evaluation metrics to {os.path.join(script_dir, 'metrics.json')}")
    print(f"Saving confusion matrix to {os.path.join(script_dir, 'confusion_matrix.png')}")
    print("-" * 80)
    print("Evaluation Metrics")
    print("-" * 80)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
