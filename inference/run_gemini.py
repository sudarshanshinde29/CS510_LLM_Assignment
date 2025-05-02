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
### Role Instruction
Act as a senior software engineer specializing in algorithmic problem solving and production-grade code implementation. Read and analyze the problem step by step. Generate a complete, optimized solution adhering strictly to the provided specifications.

### Problem Context
'''
1.⁠ ⁠Problem Description: {prob_desc_description}
2.⁠ ⁠Input Specification: {prob_desc_input_spec}
3.⁠ ⁠Output Specification: {prob_desc_output_spec}
4.⁠ ⁠Sample Cases:
   - Input: {prob_desc_sample_inputs}
   - Expected Output: {prob_desc_sample_outputs}
   - Explanation: {prob_desc_notes}
'''

### Technical Requirements
'''
•⁠  ⁠Target Language: {lang} {env_map[lang]}
•⁠  ⁠Code Constraints: Minimize external dependencies and complex headers
•⁠  ⁠Performance: Optimize for time/space complexity
•⁠  ⁠Standards: Follow {lang} best practices and PEP8/equivalent style guidelines
'''

### Output Format Specification
'''
[{{
  "version": "<exact_language_version>",
  "target_code": "<complete_solution_code>"
}}]
'''

### Generation Rules
1.⁠ ⁠Analyze sample I/O patterns to derive implementation logic
2.⁠ ⁠Validate solution against all specified edge cases
3.⁠ ⁠Include necessary standard library imports
4.⁠ ⁠Avoid unnecessary comments but maintain readable code
5.⁠ ⁠Ensure strict JSON syntax with proper escaping
6.⁠ ⁠Prohibit markdown formatting or textual explanations
7. Follow a step by step reasoning approach to ensure clarity and correctness


### Example Response
[{{
  "version": "Python 3.11",
  "target_code": "def solution(args):\n    ..."
}}]

*Critical Implementation Notes:*
•⁠  ⁠Output MUST contain ONLY valid JSON parsable by json.loads()
•⁠  ⁠Never include markdown formatting or triple backticks
•⁠  ⁠Escape all special characters properly
•⁠  ⁠Validate JSON syntax before final output

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