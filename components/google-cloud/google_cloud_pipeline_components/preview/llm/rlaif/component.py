"""Defines an RLAIF Kubeflow pipeline.

This pipeline build preference data, performs supervised fine-tuning, trains a
reward model and performs reinforcement learning.
"""

from typing import NamedTuple, Optional

from google_cloud_pipeline_components import _placeholders
from google_cloud_pipeline_components._implementation.llm import autosxs_arbiter
from google_cloud_pipeline_components._implementation.llm import preference_data_formatter
from google_cloud_pipeline_components.preview.llm.infer import component as infer
from google_cloud_pipeline_components.preview.llm.rlhf import component as rlhf
import kfp

PipelineOutput = NamedTuple(
    'Outputs', model_resource_name=str, endpoint_resource_name=str
)


@kfp.dsl.pipeline(
    name='rlaif_pipeline',
    description='Performs reinforcement learning from AI feedback.',
)
def rlaif_pipeline(
    prompt_dataset: str,
    preference_prompt_dataset: str,
    large_model_reference: str,
    model_display_name: Optional[str] = None,
    prompt_sequence_length: int = 512,
    target_sequence_length: int = 64,
    large_model_a_reference: str = 'text-bison@001',
    large_model_b_reference: str = 'elephant',
    reward_model_learning_rate_multiplier: float = 1.0,
    reinforcement_learning_rate_multiplier: float = 1.0,
    reward_model_train_steps: int = 7500,
    reinforcement_learning_train_steps: int = 5000,
    kl_coeff: float = 0.1,
    sampling_strategy: str = 'temperature_sampling',
    instruction: Optional[str] = None,
    deploy_model: bool = True,
    eval_dataset: Optional[str] = None,
    project: str = _placeholders.PROJECT_ID_PLACEHOLDER,
    location: str = _placeholders.LOCATION_PLACEHOLDER,
) -> PipelineOutput:
  """Performs reinforcement learning from human feedback.

  Args:
    prompt_dataset: Cloud storage path to an unlabled JSONL dataset that
      contains prompts. Text datasets must contain an `input_text` field that
      contains the prompt. Chat datasets must contain at least 1 message in a
      `messages` field. Each message must be valid JSON that contains `author`
      and `content` fields, where valid `author` values are `user` and
      `assistant` and `content` must be non-empty. Each row may contain multiple
      messages, but the first and last author must be the `user`. An optional
      `context` field may be provided for each example in a chat dataset. If
      provided, the `context` will preprended to the message `content`. The
      `instruction` serves as the default context. (Useful if most messages use
      the same system-level context.) Any context provided in the example will
      override the default value.
    preference_prompt_dataset: The prompt dataset used for two models'
      inferences to build the side by side comparison AI feedback.
    large_model_reference: Name of the base model. Supported values are
      `text-bison@001`, `t5-small`, `t5-large`, `t5-xl` and `t5-xxl`.
      `text-bison@001` and `t5-small` are supported in `us-central1` and
      `europe-west4`. `t5-large`, `t5-xl` and `t5-xxl` are only supported in
      `europe-west4`.
    model_display_name: Name of the fine-tuned model shown in the Model
      Registry. If not provided, a default name will be created.
    prompt_sequence_length: Maximum tokenized sequence length for input text.
      Higher values increase memory overhead. This value should be at most 8192.
      Default value is 512.
    target_sequence_length:  Maximum tokenized sequence length for target text.
      Higher values increase memory overhead. This value should be at most 1024.
      Default value is 64.
    large_model_a_reference: Name of a predefined model A for side by side
      comparison to build the AI feedback dataset. By default, it uses
      `text-bison@001`. The valid values are `T5_SMALL`, `T5_LARGE`, `T5_XL`,
      `T5_XXL`, `GECKO`, `OTTER`, `text-bison@001`, `ELEPHANT`.
    large_model_b_reference: Name of a predefined model B for side by side
      comparison to build the AI feedback dataset. By default, it uses
      `ELEPHANT`. The valid values are `T5_SMALL`, `T5_LARGE`, `T5_XL`,
      `T5_XXL`, `GECKO`, `OTTER`, `text-bison@001`, `ELEPHANT`.
    reward_model_learning_rate_multiplier: Constant used to adjust the base
      learning rate used when training a reward model. Multiply by a number > 1
      to increase the magnitude of updates applied at each training step or
      multiply by a number < 1 to decrease the magnitude of updates. Default
      value is 1.0.
    reinforcement_learning_rate_multiplier: Constant used to adjust the base
      learning rate used during reinforcement learning. Multiply by a number > 1
      to increase the magnitude of updates applied at each training step or
      multiply by a number < 1 to decrease the magnitude of updates. Default
      value is 1.0.
    reward_model_train_steps: Number of steps to use when training a reward
      model. Default value is 1000.
    reinforcement_learning_train_steps: Number of reinforcement learning steps
      to perform when tuning a base model. Default value is 1000.
    kl_coeff: Coefficient for KL penalty. This regularizes the policy model and
      penalizes if it diverges from its initial distribution. If set to 0, the
      reference language model is not loaded into memory. Default value is 0.1.
    sampling_strategy: The strategy used to candidates for AI feedback. Default
      is temperature_sampling. Valid values are greedy, temperature_sampling
    instruction: This field lets the model know what task it needs to perform.
      Base models have been trained over a large set of varied instructions. You
      can give a simple and intuitive description of the task and the model will
      follow it, e.g., "Classify this movie review as positive or negative" or
      "Translate this sentence to Danish". Do not specify this if your dataset
      already prepends the instruction to the inputs field.
    deploy_model: Whether to deploy the model to an endpoint in `us-central1`.
      Default is True.
    eval_dataset: Optional Cloud storage path to an evaluation dataset. If
      provided, inference will be performed on this dataset after training. The
      dataset format is jsonl. Each example in the dataset must contain a field
      `input_text` that contains the prompt.
    project: Project used to run custom jobs. If not specified the project used
      to run the pipeline will be used.
    location: Location used to run custom jobs. If not specified the location
      used to run the pipeline will be used.
    tensorboard_resource_id: Optional tensorboard resource id in format
      `projects/{project_number}/locations/{location}/tensorboards/{tensorboard_id}`.
      If provided, tensorboard metrics will be uploaded to this location.

  Returns:
    model_resource_name: Path to the model uploaded to the Model Registry. This
    will be an empty string if the model was not deployed.
    endpoint_resource_name: Path the Online Prediction Endpoint. This will be an
    empty string if the model was not deployed.
  """
  id_columns = ['content']
  task = 'summarization@001'

  output_prediction_gcs_path_a = infer.infer_pipeline(
      large_model_reference=large_model_a_reference,
      prompt_dataset=preference_prompt_dataset,
      prompt_sequence_length=prompt_sequence_length,
      target_sequence_length=target_sequence_length,
      sampling_strategy=sampling_strategy,
      instruction=instruction,
      project=project,
      location=location,
  ).set_display_name('Inferrer A')
  output_prediction_gcs_path_b = infer.infer_pipeline(
      large_model_reference=large_model_b_reference,
      prompt_dataset=preference_prompt_dataset,
      prompt_sequence_length=prompt_sequence_length,
      target_sequence_length=target_sequence_length,
      sampling_strategy=sampling_strategy,
      instruction=instruction,
      project=project,
      location=location,
  ).set_display_name('Inferrer B')

  inference_output_uri = (
      preference_data_formatter.format_preference_input_data(
          model_a_inference_dir_uri=output_prediction_gcs_path_a.output,
          model_b_inference_dir_uri=output_prediction_gcs_path_b.output,
          instruction=instruction,
      )
      .set_display_name('Prepare AI Feedback Input')
      .output
  )

  autosxs = autosxs_arbiter.autosxs_arbiter(
      inference_output_uri=inference_output_uri,
      id_columns=id_columns,
      task=task,
  ).set_display_name('Build AI Feedback')

  preference_dataset = (
      preference_data_formatter.format_preference_data(
          input_uri=autosxs.outputs['judgments_uri']
      )
      .set_display_name('Build Preference Dataset')
      .output
  )

  rlhf_outputs = (
      rlhf.rlhf_pipeline(
          prompt_dataset=prompt_dataset,
          preference_dataset=preference_dataset,
          large_model_reference=large_model_reference,
          model_display_name=model_display_name,
          prompt_sequence_length=prompt_sequence_length,
          target_sequence_length=target_sequence_length,
          reward_model_train_steps=reward_model_train_steps,
          reinforcement_learning_train_steps=reinforcement_learning_train_steps,
          reward_model_learning_rate_multiplier=reward_model_learning_rate_multiplier,
          reinforcement_learning_rate_multiplier=reinforcement_learning_rate_multiplier,
          instruction=instruction,
          deploy_model=deploy_model,
          eval_dataset=eval_dataset,
          kl_coeff=kl_coeff,
          project=project,
          location=location,
      )
      .set_display_name('Reinforcement Learning From AI Feedback')
      .outputs
  )
  return PipelineOutput(
      model_resource_name=rlhf_outputs['model_resource_name'],
      endpoint_resource_name=rlhf_outputs['endpoint_resource_name'],
  )
