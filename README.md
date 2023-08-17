# OpenAI API를 사용한 토큰 계산 및 병렬 처리 스크립트
이 스크립트는 OpenAI API를 사용하여 텍스트 파일의 토큰을 계산하고, 주어진 프롬프트에 대해 OpenAI 모델에 병렬로 요청을 보낸 뒤 결과를 하나로 모아 저장합니다.

## 만든 이유
- 업무중 대량의 데이터를 일괄적으로 변경할 필요가 있었습니다. 그런데 코드 짜기 난해한 경우도 있어서 이런 경우 GPT 모델을 이용했습니다. 그런데 이 방법마저도 간혹 token 수 제한 때문에 요청을 여러번 나눠보내야 하는 어려움이 있었습니다.
- 집에서 쉬고 있는데 발더스게이트3가 출시된지 하루만에 한글패치가 배포되어서 어떻게 했을까 궁금했습니다. 이야기를 들어보니 AI(ChatGPT)를 이용해 번역을 했다고 듣게되었고 아마도 번역팀에 개발자가 있다면 이런 형태로 하지 않았을까? 싶어 직접 해보고 싶었습니다.
- KARS에 i18n이 적용되었는데도 불구하고 번역 해 줄 사람이 없어서 다국어를 지원 못하고 있는 현실이 안타까웠습니다.
- 파이썬의 poetry와 typer를 사용해보고 싶었습니다.

## 사용한 라이브러리

```text
json
math
os
concurrent.futures
openai
tiktoken
typer
rich
tqdm
```

## 사용법
1. 프롬프트를 입력합니다.
    ```bash
    poetry run python main.py
    ```
2. 대상 파일 이름을 입력합니다.
3. OpenAI API 키를 입력합니다. (config.json 파일에 기본값이 있으면 그것을 사용할 수 있습니다.)
4. 모델 이름을 입력합니다. (config.json 파일에 기본값이 있으면 그것을 사용할 수 있습니다.)
5. 계산 결과를 확인합니다.
6. 결과는 "result_" + 파일명에 저장됩니다.

## 주요 함수
- count_tokens(string: str, encoding_name: str) -> int: 주어진 문자열의 토큰 수를 계산합니다.
- get_file_info(string: str, encoding_name: str) -> dict: 파일의 토큰 수, 평균 토큰 수, 총 라인 수를 계산합니다.
- ask_openai(model_name, system_message, user_message): OpenAI에 질문을 하고 응답을 받습니다.
- ask_openai_parallel(model_name, system_user_pairs, output_file): 여러 질문을 병렬로 처리하고 결과를 파일에 저장합니다.
- create_pairs_from_file(file, prompt, lines_per_pair): 파일에서 질문을 생성합니다.

## 주의 사항
- 사용 가능한 토큰 수는 모델에 따라 다르며, 이 스크립트는 각 모델의 최대 토큰 수를 미리 정의하고 있습니다.
- OpenAI API 키는 개인 정보이므로 보안에 주의해야 합니다. 이 스크립트는 API 키를 config.json 파일에 저장합니다.
- 이 프로그램은 Python 3.7 이상에서 작동합니다.

## 활용 사례
- 대량의 텍스트 데이터 처리: 이 프로그램은 대량의 텍스트 데이터를 처리하는데 유용합니다. 텍스트 파일을 읽어서 OpenAI API에 요청을 보낼 수 있으므로, 대량의 텍스트 데이터를 분석하거나 처리해야 하는 경우에 사용할 수 있습니다.
- 토큰 수 계산: OpenAI API를 사용할 때 토큰 수는 중요한 요소입니다. 이 프로그램은 텍스트의 토큰 수를 계산하여 API 요청 시 필요한 토큰 수를 예측할 수 있게 도와줍니다.
- API 요청 자동화: 이 프로그램은 OpenAI API에 요청을 보내는 과정을 자동화합니다. 사용자는 프롬프트와 파일 이름만 입력하면 프로그램이 나머지 작업을 처리합니다. 이를 통해 API 요청 과정을 간소화하고 시간을 절약할 수 있습니다.
- 병렬 처리: 이 프로그램은 병렬 처리를 지원하여 여러 요청을 동시에 처리할 수 있습니다. 이를 통해 대량의 데이터를 빠르게 처리할 수 있습니다.

