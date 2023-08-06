import argparse
import math
from concurrent.futures import ThreadPoolExecutor

import openai
import tiktoken
from tqdm import tqdm

# MODEL_NAME = "gpt-3.5-turbo-16k-0613"
# MAX_TOKEN = 4096 * 4
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


# 파싱된 인자 출력
def print_args(params):
    for param in params.inputs:
        print(param)


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

        return {"count_tokens": count_tokens, "average": average, "total_line": total_line}


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


def create_pairs_from_file(file_name, prompt, lines_per_pair):
    with open(file_name, 'r') as file:
        content = file.read().splitlines()

    pairs = [(prompt, ' '.join(content[i:i + lines_per_pair])) for i in range(0, len(content), lines_per_pair)]

    return pairs


parser = argparse.ArgumentParser(description="This is a simple argument parser")

# 인자 추가
parser.add_argument("inputs", type=str, nargs='+', help="Input to be printed out")

# 인자 파싱
args = parser.parse_args()
openai.api_key = args.inputs[2]

if __name__ == '__main__':
    print_args(args)

    prompt_token = count_tokens(args.inputs[0], args.inputs[1])
    print("# prompt token : " + str(prompt_token))

    file_info = get_file_info(args.inputs[3], args.inputs[1])

    print("# file token : " + str(file_info))

    available_token = math.floor((MAX_TOKENS[args.inputs[1]] - prompt_token) / 2 - BUFFER)
    print("# available token : " + str(available_token))

    line_per_request = math.floor(available_token / file_info.get("average"))
    print("# line per request : " + str(line_per_request))

    total_request = math.ceil(file_info.get("total_line") / line_per_request)
    print("# total request : " + str(total_request))

    pairs = create_pairs_from_file(args.inputs[3], args.inputs[0], line_per_request)

    ask_openai_parallel(args.inputs[1], pairs, "result_" + args.inputs[3])