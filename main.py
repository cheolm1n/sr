import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import openai
import tiktoken
import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

BUFFER = 50
MAX_TOKENS = {
    "gpt-4": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0613": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k-0613": 16384,
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    "code-davinci-002": 8001,
    "text-curie-001": 2049,
    "text-babbage-001": 2049,
    "text-ada-001": 2049,
    "davinci": 2049,
    "curie": 2049,
    "babbage": 2049,
    "ada": 2049
}


def count_tokens(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_file_info(string: str, encoding_name: str) -> dict:
    with open(string) as file:
        lines = file.readlines()

        text = ''.join(lines)

        cnt_tokens = count_tokens(text, encoding_name)

        total_line = len(lines)

        average = math.ceil(cnt_tokens / total_line)

        return {"count_tokens": cnt_tokens, "average": average, "total_line": total_line}


def ask_openai(model_name, system_message, user_message):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    )
    return response['choices'][0]['message']['content']


def ask_openai_parallel(model_name, system_user_pairs, output_file):
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(ask_openai, model_name, pair[0], pair[1]): i for i, pair in enumerate(system_user_pairs)}
        results = [None] * len(futures)

        with tqdm(total=len(futures), desc="Processing pairs", ncols=70) as pbar:
            for future in futures:
                results[futures[future]] = future.result()
                pbar.update()

        with open(output_file, 'w') as file:
            for result in results:
                file.write(result + '\n')


def create_pairs_from_file(file, prompt, lines_per_pair):
    with open(file, 'r') as file:
        content = file.read().splitlines()

    pairs = [(prompt, ' '.join(content[i:i + lines_per_pair])) for i in range(0, len(content), lines_per_pair)]

    return pairs


console = Console()


def main():
    prompt = typer.prompt("Enter prompt")
    file = typer.prompt("Enter the target file name")

    if os.path.exists('config.json'):
        with open('config.json') as json_file:
            data = json.load(json_file)
            default_api_key = data['api_key']
            default_model_name = data.get('model_name', "gpt-3.5-turbo-16k-0613")
    else:
        default_api_key = None
        default_model_name = "gpt-3.5-turbo-16k-0613"

    api_key = typer.prompt("Enter your api key for OpenAI", default=default_api_key)
    model_name = typer.prompt("Enter the model name", default=default_model_name)

    with open('config.json', 'w') as outfile:
        json.dump({'api_key': api_key, 'model_name': model_name}, outfile)

    openai.api_key = api_key

    prompt_token = count_tokens(prompt, model_name)
    file_info = get_file_info(file, model_name)
    available_token = math.floor((MAX_TOKENS[model_name] - prompt_token) / 2 - BUFFER)
    line_per_request = math.floor(available_token / file_info.get("average"))
    total_request = math.ceil(file_info.get("total_line") / line_per_request)

    print("\n")
    print("Here's the result of the calculation.")
    table = Table("Name", "Value")
    table.add_row("prompt token", str(prompt_token))
    table.add_row("file token", str(file_info["count_tokens"]))
    table.add_row("total line", str(file_info["total_line"]))
    table.add_row("average", str(file_info["average"]))
    table.add_row("available token", str(available_token))
    table.add_row("line per request", str(line_per_request))
    table.add_row(Text("total request", style="red"), str(total_request))
    console.print(table)
    print("\n")

    pairs = create_pairs_from_file(file, prompt, line_per_request)

    typer.confirm("Do you want to run?", abort=True)

    ask_openai_parallel(model_name, pairs, "result_" + file)


if __name__ == "__main__":
    typer.run(main)
