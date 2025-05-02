import time
from google import genai
import backoff
import logging
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset
from google.genai import types
import json
from google.api_core.exceptions import GoogleAPIError

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', required=True, type=str)
    parser.add_argument('--model', default='gemini-2.0-flash', choices=['gemini-2.0-flash'], type=str)
    parser.add_argument('--data_load_name', default='program_synthesis_data.jsonl', type=str)
    parser.add_argument('--result_save_name', default='program_synthesis_eval_gemini.jsonl', type=str)
    parser.add_argument('--log_file_name', default='program_synthesis_eval_gemini.logs', type=str)
    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--candidate_num', default=2, type=int)
    args = parser.parse_args()
    return args

def is_503_error(e):
    return hasattr(e, 'code') and e.code == 503

@backoff.on_exception(
    backoff.expo,
    GoogleAPIError,
    giveup=lambda e: not is_503_error(e),
    max_tries=10
)
def generate_text(client, model, prompt, temperature, candidate_num):
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=temperature,
            candidate_count=candidate_num,
        )
    )
    results = [candidate.content.parts[0].text for candidate in response.candidates]

    return results

env_map = {
    'python': ['python2', 'python3', ],
}

lang_cluster = ['python']

def add_program_synthesis(example, client):
    prob_uid = example['src_uid']
    prob_desc_description = example['description']
    prob_desc_input_spec = example['input_specification']
    prob_desc_output_spec = example['output_specification']
    prob_desc_sample_inputs = example['sample_inputs']
    prob_desc_sample_outputs = example['sample_outputs']
    prob_desc_notes = example['notes']
    lang = example['lang_cluster'].lower()

    prompt = f"""
You are a dependable and precise code generation assistant. Stick closely to the structure and constraints provided. Think carefully before coding.
---

## Role Definition
Act as an experienced software engineer and technical specialist. Follow instructions exactly.

---

## Objective
Write valid code that strictly adheres to the problem statement provided in the ⁠ TASK DETAILS ⁠. The output should follow the structure defined in the ⁠ RESPONSE FORMAT⁠  section. Avoid any extra text or commentary.

### TASK DETAILS
'''
1.⁠ ⁠Task Overview: {prob_desc_description}
2.⁠ ⁠Input Format: {prob_desc_input_spec}
3.⁠ ⁠Output Requirements: {prob_desc_output_spec}
4.⁠ ⁠Examples:
   - Sample Input: {prob_desc_sample_inputs}
   - Sample Output: {prob_desc_sample_outputs}
   - Notes: {prob_desc_notes}
5.⁠ ⁠Language: {lang} {env_map[lang]}
6.⁠ ⁠Coding Guidelines: Follow best practices and style conventions for {lang}
'''

---

## Execution Rules
•⁠  ⁠Parse and understand the task carefully before starting.
•⁠  ⁠Provide a brief thought process first.
•⁠  ⁠Output only the final code, enclosed in a code block, with no explanation outside it.

---

## RESPONSE FORMAT
'''
[{{
  "version": "<programming_language_version>",
  "target_code": "<fully working solution>"
}}]
'''

### Template Response
[{{
  "version": "Python 3.11",
  "target_code": "def solve(...):\\n    # implementation here"
}}]
"""

    logging.info('problem src_id: ' + str(prob_uid))
    logging.info(prompt)

    try:
        response = generate_text(
            client=client,
            model=args.model,
            prompt=prompt,
            temperature=temperature,
            candidate_num=candidate_num
        )
        logging.info('response: ' + str(response))
        time.sleep(25)

        if response is not None:
            program_sythesis = response
        else:
            logging.warning('Respond content is none.')
            program_sythesis = []

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        program_sythesis = []

    logging.info('program_synthesis in: ' + lang + ' :' + str(program_sythesis))
    example['program_synthesis'] = program_sythesis

    for i, generated_code in enumerate(program_sythesis):
        logging.info('program_synthesis  in: ' + lang + ' :' + generated_code)
        example['program_synthesis_' + str(i)] = generated_code

    if len(program_sythesis) < candidate_num:
        for i in range(candidate_num - len(program_sythesis)):
            example['program_synthesis_' + str(i + len(program_sythesis))] = ''

    return example

def main(client):
    load_path = Path(__file__).parent.parent / Path('data') / Path(args.data_load_name)
    save_path = Path(__file__).parent / Path('results') / Path(args.result_save_name)

    dataset = load_dataset('json', split='train', data_files=str(load_path))
    dataset.cleanup_cache_files()

    dataset = dataset.map(lambda example: add_program_synthesis(example, client))

    dataset.to_json(save_path, lines=True)

if __name__ == '__main__':
    args = parse_arguments()

    log_file_path = Path(__file__).parent / Path('logs') / Path(args.log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(filename)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filename=log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    client = genai.Client(api_key=args.api_key) # Initialize with API key

    candidate_num = args.candidate_num
    temperature = args.temperature

    main(client)