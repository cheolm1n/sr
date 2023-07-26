import argparse

import tiktoken

MODEL_NAME = "gpt-3.5-turbo-16k-0613"
MAX_TOKEN = 4096 * 4
BUFFER = 50


# 파싱된 인자 출력
def print_args(params):
    for param in params.inputs:
        print(param)


def count_from_file(string: str, encoding_name: str) -> int:
    with open(string) as file:
        text = file.read()

        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(text))

        average_10_lines = len(file.readlines()[:10])

        total_line = len(file.readlines())

        return {"num_tokens": num_tokens, "average_10_lines": average_10_lines, "total_line": total_line}


def count_tokens(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


parser = argparse.ArgumentParser(description="This is a simple argument parser")

# 인자 추가
parser.add_argument("inputs", type=str, nargs='+', help="Input to be printed out")

# 인자 파싱
args = parser.parse_args()

if __name__ == '__main__':
    print_args(args)
    prompt_token = count_tokens(args.inputs[0], MODEL_NAME)
    print("prompt token : " + str(prompt_token))

    count_dict = count_from_file(args.inputs[3], MODEL_NAME)

    total_file_token =count_dict.
    print("total file token : " + str(total_file_token))

    available_token = (MAX_TOKEN - prompt_token) / 2 - BUFFER
    print("available token : " + str(available_token))

    request_count = available_token /

# prompt
# model
# api key
# file path

# available : (model_max_token - prompt_token_count) / 2 - buffer
# request_count : total_line / available / average_token_per_line
