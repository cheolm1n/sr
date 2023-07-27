import argparse
import math

import tiktoken

MODEL_NAME = "gpt-3.5-turbo-16k-0613"
MAX_TOKEN = 4096 * 4
BUFFER = 50


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


parser = argparse.ArgumentParser(description="This is a simple argument parser")

# 인자 추가
parser.add_argument("inputs", type=str, nargs='+', help="Input to be printed out")

# 인자 파싱
args = parser.parse_args()

if __name__ == '__main__':
    print_args(args)

    prompt_token = count_tokens(args.inputs[0], MODEL_NAME)
    print("# prompt token : " + str(prompt_token))

    file_info = get_file_info(args.inputs[3], MODEL_NAME)

    print("# file token : " + str(file_info))

    available_token = math.floor((MAX_TOKEN - prompt_token) / 2 - BUFFER)
    print("# available token : " + str(available_token))

    line_per_request = math.floor(available_token / file_info.get("average"))
    print("# line per request : " + str(line_per_request))

    total_request = math.ceil(file_info.get("total_line") / line_per_request)
    print("# total request : " + str(total_request))

# prompt
# model
# api key
# file path

# available : (model_max_token - prompt_token_count) / 2 - buffer
# request_count : total_line / available / average_token_per_line
