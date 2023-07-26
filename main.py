import argparse


# 파싱된 인자 출력
def print_hi(args):
    for input in args.inputs:
        print(input)


parser = argparse.ArgumentParser(description="This is a simple argument parser")

# 인자 추가
parser.add_argument("inputs", type=str, nargs='+', help="Input to be printed out")

# 인자 파싱
args = parser.parse_args()


if __name__ == '__main__':
    print_hi(args)
