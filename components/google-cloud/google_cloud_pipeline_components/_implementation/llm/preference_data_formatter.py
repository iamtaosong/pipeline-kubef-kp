"""Utility function to format the preference data."""

from typing import Any, Dict, List
from kfp import dsl

BASE_IMAGE = 'us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-8:latest'


# pylint: disable=g-import-not-at-top
@dsl.component(
    base_image=BASE_IMAGE,
    # TODO(b/288343729): build our own base image and use setting
    # install_kfp_package=False
)
def group_model_a_inference_candidates(
    input_dir_uri: str,
    num_of_samples_per_group: int = 500,
) -> Dict[str, Any]:
  """Group the model A inference data into multiple groups.

  Args:
    input_dir_uri: Where the inference data was saved in the previous step
      infer_pipeline run.
    num_of_samples_per_group: The number of samples in each group.

  Returns:
    The list of directory pathes of the candidate groups.
  """
  import tensorflow as tf
  import json
  import os
  import hashlib
  import math

  output_uri_list = []
  input_data_lines = []
  for filename in tf.io.gfile.glob(os.path.join(input_dir_uri, '*')):
    with tf.io.gfile.GFile(filename, 'r') as f:
      input_data_lines.extend(f.readlines())

  num_of_groups = math.ceil(len(input_data_lines) / num_of_samples_per_group)
  # The hash map for grouping the input data lines. For the key, it's
  # group_id(str) generated based on the prompt's 'inputs' field(excluding the
  # prediction results). For the value, it's a list of the data lines that
  # belong to this group.
  group_map = {}
  hash_value_to_group_id_map = {}

  for line in input_data_lines:
    line_json = json.loads(line)
    # Have to use hashlib to ensure same value for same input string. The hash()
    # function returns different values in two different components(Grouping
    # dataset A vs grouping dataset B).
    hash_obj = hashlib.md5(
        json.dumps(line_json['inputs']['inputs_pretokenized']).encode()
    )
    hash_int = int(hash_obj.hexdigest(), 16)
    hash_value_str = str(hash_int)
    group_id = hash_int % num_of_groups
    if str(group_id) not in group_map:
      group_map[str(group_id)] = []
    # Find the next available group once the current group is full.
    while len(group_map[str(group_id)]) >= num_of_samples_per_group:
      group_id = (group_id + 1) % num_of_groups
    hash_value_to_group_id_map[hash_value_str] = str(group_id)
    group_map[str(group_id)].append(line)

  output_dir_uri = os.path.join(input_dir_uri, 'candidate_groups')
  tf.io.gfile.makedirs(
      os.path.dirname(output_dir_uri),
  )
  for group_num, candidate_group in group_map.items():
    group_output_dir_uri = os.path.join(output_dir_uri, f'groups_{group_num}')
    tf.io.gfile.makedirs(
        os.path.dirname(group_output_dir_uri),
    )
    group_output_uri = os.path.join(
        group_output_dir_uri, f'candidate_group_{group_num}.jsonl'
    )
    with tf.io.gfile.GFile(group_output_uri, 'w') as f:
      for line in candidate_group:
        f.write(line)
    output_uri_list.append(group_output_dir_uri)
  # output hashmap
  hash_map_output_uri = os.path.join(
      output_dir_uri, 'hash_value_to_group_id_map.jsonl'
  )
  with tf.io.gfile.GFile(hash_map_output_uri, 'w') as f:
    f.write(json.dumps(hash_value_to_group_id_map))
  return {
      'candidate_group_uri_list': output_uri_list,
      'hash_value_to_group_id_map_uri': hash_map_output_uri,
  }


# pylint: disable=g-import-not-at-top
@dsl.component(
    base_image=BASE_IMAGE,
    # TODO(b/288343729): build our own base image and use setting
    # install_kfp_package=False
)
def group_model_b_inference_candidates(
    input_dir_uri: str,
    grouping_model_a_candidates_results: Dict[str, Any],
) -> List[str]:
  """Group the model B inference data into multiple groups.

  Args:
    input_dir_uri: Where the inference data was saved in the previous step
      infer_pipeline run.
    grouping_model_a_candidates_results: The hash map to store the metadata from
      prevous step of grouping the model A candidates. The
      hash_value_to_group_id_map inside will be used to track which group each
      line belongs to.

  Returns:
    The list of directory pathes of the candidate groups.
  """
  import tensorflow as tf
  import json
  import os
  import hashlib

  hash_value_to_group_id_map_uri = grouping_model_a_candidates_results[
      'hash_value_to_group_id_map_uri'
  ]
  hash_value_to_group_id_map = {}
  with tf.io.gfile.GFile(hash_value_to_group_id_map_uri, 'r') as f:
    hash_value_to_group_id_map.update(json.load(f))

  output_uri_list = []
  input_data_lines = []
  for filename in tf.io.gfile.glob(os.path.join(input_dir_uri, '*')):
    with tf.io.gfile.GFile(filename, 'r') as f:
      input_data_lines.extend(f.readlines())

  # The hash map for grouping the input data lines. For the key, it's
  # group_id(str) generated based on the prompt's 'inputs' field(excluding the
  # prediction results). For the value, it's a list of the data lines that
  # belong to this group.
  group_map = {}

  for line in input_data_lines:
    line_json = json.loads(line)
    # Have to use hashlib to ensure same value for same input string. The hash()
    # function returns different values in two different components(Grouping
    # dataset A vs grouping dataset B).
    hash_obj = hashlib.md5(
        json.dumps(line_json['inputs']['inputs_pretokenized']).encode()
    )
    hash_int = int(hash_obj.hexdigest(), 16)
    hash_value_str = str(hash_int)
    group_id_str = hash_value_to_group_id_map[hash_value_str]
    if group_id_str not in group_map:
      group_map[group_id_str] = []
    group_map[group_id_str].append(line)

  output_dir_uri = os.path.join(input_dir_uri, 'candidate_groups')
  tf.io.gfile.makedirs(
      os.path.dirname(output_dir_uri),
  )
  for group_num, candidate_group in group_map.items():
    group_output_dir_uri = os.path.join(output_dir_uri, f'groups_{group_num}')
    tf.io.gfile.makedirs(
        os.path.dirname(group_output_dir_uri),
    )
    group_output_uri = os.path.join(
        group_output_dir_uri, f'candidate_group_{group_num}.jsonl'
    )
    with tf.io.gfile.GFile(group_output_uri, 'w') as f:
      for line in candidate_group:
        f.write(line)
    output_uri_list.append(group_output_dir_uri)
  return output_uri_list


# pylint: disable=g-import-not-at-top
@dsl.component(
    base_image=BASE_IMAGE,
    # TODO(b/288343729): build our own base image and use setting
    # install_kfp_package=False
)
def build_candidate_group_pair(
    grouping_model_a_candidates_results: Dict[str, Any],
    inference_b_candidate_groups: List[str],
) -> list:  # pylint: disable=g-bare-generic
  """Build the candidate group pair.

  Args:
    grouping_model_a_candidates_results: The hash map to store the metadata from
      prevous step of grouping the model A candidates. The field
      inference_a_candidate_groups is used to store the paths to all inference A
      groups.
    inference_b_candidate_groups: paths to inference B groups.

  Returns:
    List inference group pairs for comparisons.
  """
  candidate_group_pair_list = []
  inference_a_candidate_groups = grouping_model_a_candidates_results[
      'candidate_group_uri_list'
  ]
  assert len(inference_a_candidate_groups) == len(inference_b_candidate_groups)
  # Ensure the list is ordered in '/group_0/', '/group_1/'... This ensures the
  # match between the dataset A list and dataset B list.
  inference_a_candidate_groups.sort()
  inference_b_candidate_groups.sort()
  for group_pair in list(
      zip(inference_a_candidate_groups, inference_b_candidate_groups)
  ):
    inference_a_candidate_group, inference_b_candidate_group = group_pair
    pair = {
        'inference_a': inference_a_candidate_group,
        'inference_b': inference_b_candidate_group,
    }
    candidate_group_pair_list.append(pair)
  return candidate_group_pair_list


# pylint: disable=g-import-not-at-top
@dsl.component(
    base_image=BASE_IMAGE,
    # TODO(b/288343729): build our own base image and use setting
    # install_kfp_package=False
)
def merge_comparisons(comparison_paths: List[str]) -> str:
  """Merge evaluations.

  Args:
    comparison_paths: paths to evaluations.

  Returns:
    Path to merged evaluations.
  """
  import tensorflow as tf

  output_uri = ''
  input_data_lines = []
  for filename in tf.io.gfile.glob(comparison_paths):
    with tf.io.gfile.GFile(filename, 'r') as f:
      input_data_lines.extend(f.readlines())
    output_uri = filename.replace('.jsonl', '_merged.jsonl')

  with tf.io.gfile.GFile(output_uri, 'w') as f:
    for line in input_data_lines:
      f.write(line)
  return output_uri


# pylint: disable=g-import-not-at-top
@dsl.component(
    base_image=BASE_IMAGE,
    # TODO(b/288343729): build our own base image and use setting
    # install_kfp_package=False
)
def format_preference_input_data(
    model_a_inference_dir_uri: str,
    model_b_inference_dir_uri: str,
    instruction: str,
) -> str:
  """Format the inference data from model a and model b and merge them as the input for auto sxs evaluation.

  Args:
    model_a_inference_dir_uri: Where the model a judgments data was saved in the
      previous step.
    model_b_inference_dir_uri: Where the model b judgments data was saved in the
      previous step.
    instruction: instruction to the task.

  Returns:
    The path to the new output file that saved the formatted input data for
    AutoSxs arbiter.
  """
  import tensorflow as tf
  import json
  import hashlib
  import os

  model_a_inference_data_map = {}
  model_b_inference_data_map = {}
  files_in_folder_a = tf.io.gfile.glob(
      os.path.join(model_a_inference_dir_uri, 'text*')
  )
  files_in_folder_b = tf.io.gfile.glob(
      os.path.join(model_b_inference_dir_uri, 'text*')
  )
  assert (
      len(files_in_folder_a) == 1 & len(files_in_folder_b) == 1
  ), 'There should be one inference data file for each model'
  with tf.io.gfile.GFile(files_in_folder_a[0], 'r') as inputs:
    for line in inputs:
      line_json = json.loads(line)
      hash_obj = hashlib.md5(
          json.dumps(line_json['inputs']['inputs_pretokenized']).encode()
      )
      hash_int = int(hash_obj.hexdigest(), 16)
      model_a_inference_data_map[str(hash_int)] = line_json

  with tf.io.gfile.GFile(files_in_folder_b[0], 'r') as inputs:
    for line in inputs:
      line_json = json.loads(line)
      hash_obj = hashlib.md5(
          json.dumps(line_json['inputs']['inputs_pretokenized']).encode()
      )
      hash_int = int(hash_obj.hexdigest(), 16)
      model_b_inference_data_map[str(hash_int)] = line_json

  formatted_data_json = []
  for key, model_a_inference_item in model_a_inference_data_map.items():
    if key in model_b_inference_data_map:
      model_b_inference_item = model_b_inference_data_map[key]
      updated_line_json = {}
      updated_line_json['inference_instruction'] = instruction
      updated_line_json['content'] = model_a_inference_item['inputs'][
          'inputs_pretokenized'
      ]
      updated_line_json['inference_context'] = model_a_inference_item['inputs'][
          'inputs_pretokenized'
      ]
      updated_line_json['response_a'] = model_a_inference_item['prediction']
      updated_line_json['response_b'] = model_b_inference_item['prediction']
      formatted_data_json.append(updated_line_json)

  output_uri = files_in_folder_a[0].replace(
      '.jsonl', '_formatted_for_autosxs.jsonl'
  )
  with tf.io.gfile.GFile(output_uri, 'w') as f:
    for line in formatted_data_json:
      f.write(json.dumps(line))
      f.write('\n')
  return output_uri


# pylint: disable=g-import-not-at-top
@dsl.component(
    base_image=BASE_IMAGE,
    # TODO(b/288343729): build our own base image and use setting
    # install_kfp_package=False
)
def format_preference_data(input_uri: str) -> str:
  """Format the input for preference data.

  Args:
    input_uri: Where the judgments data was saved in the previous step.

  Returns:
    The path to the new output file that saved the formatted preference data.
    It's under the same folder as the original data file.
  """
  import tensorflow as tf
  import json

  output_uri = input_uri.replace('.jsonl', '_formatted_for_rlaif.jsonl')
  formatted_data_json = []
  with tf.io.gfile.GFile(input_uri, 'r') as inputs:
    for line in inputs:
      line_json = json.loads(line)
      if line_json['choice'] not in ['A', 'B']:
        continue
      updated_line_json = {}
      updated_line_json['input_text'] = line_json[
          'autorater_prompt_parameters'
      ]['inference_context']
      updated_line_json['candidate_0'] = line_json['response_a']
      updated_line_json['candidate_1'] = line_json['response_b']
      updated_line_json['choice'] = 0 if line_json['choice'] == 'A' else 1
      formatted_data_json.append(updated_line_json)

  with tf.io.gfile.GFile(output_uri, 'w') as f:
    for line in formatted_data_json:
      f.write(json.dumps(line))
      f.write('\n')
  return output_uri
