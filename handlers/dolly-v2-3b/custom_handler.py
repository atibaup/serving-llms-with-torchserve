#
# This TochServe Custom handler is adapted from the pipeline provided by the model's authors in:
# https://huggingface.co/databricks/dolly-v2-12b/tree/b7fbc6d46abb330670a97eb5f8af2c78fb868cfd
# See more info here:
# https://huggingface.co/databricks/dolly-v2-12b#usage
#
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from ts.torch_handler.base_handler import BaseHandler
import torch
import numpy as np

import logging
from typing import List, Union, Dict
import time

logger = logging.getLogger(__name__)


INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.

    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.

    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token

    Raises:
        RuntimeError: if more than one ID was generated

    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]


class InstructionTextGenerationHandler(BaseHandler):
    """
    The handler takes an input string and returns a string as an ouput
    """

    def __init__(self, do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0):
        super().__init__()
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.initialized = False

    def initialize(self, context):
        """
        Loads the model.pt file and initializes the model object.
        Instantiates Tokenizer for preprocessor to use

        Args :
            context (context): a JSON Object containing information
                pertaining to the model artifacts parameters.
        """
        self.context = context
        self.manifest = self.context.manifest
        properties = self.context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            offload_folder="offload",
            offload_state_dict=True
        )
        self.model.eval()
        logger.debug('InstructionTextGeneration model from path {} loaded successfully'.format(model_dir))

        self.response_key_token_id = get_special_token_id(self.tokenizer, RESPONSE_KEY)
        self.end_key_token_id = get_special_token_id(self.tokenizer, END_KEY)

        self.initialized = True

    def preprocess(self, instruction_text: List[str]) -> Dict[str, Union[str, torch.Tensor]]:
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing

        Args :
            instruction_text (list): List of the data from the request input.

        Returns:
            tensor: Returns the tensor data of the input
        """
        if isinstance(instruction_text, list):
            if len(instruction_text) > 1:
                raise ValueError("Can only accepts inputs that are a list of one element")
        else:
            raise ValueError("Input should be a list but is not")
        prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction_text[0])
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        return inputs

    def inference(self, model_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        generated_sequence = self.model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.end_key_token_id,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k
        )[0].cpu()
        return generated_sequence

    def postprocess(self, model_outputs: torch.Tensor) -> List[str]:
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.

        Args:
            model_outputs (Torch Tensor): The torch tensor received from the prediction output of the model.

        Returns:
            List: The post process function returns a list of the predicted output.
        """
        logger.info(f"Raw decode:\n****************************\n{self.tokenizer.decode(model_outputs).strip()}\n****************************")

        # The response will be set to this variable if we can identify it.
        decoded = None

        # Find where "### Response:" is first found in the generated tokens.  Considering this is part of the
        # prompt, we should definitely find it.  We will return the tokens found after this token.
        response_pos = None
        response_positions = np.where(model_outputs == self.response_key_token_id)[0]
        if len(response_positions) == 0:
            logger.warning(f"Could not find response key {self.response_key_token_id} in: {model_outputs}")
        else:
            response_pos = response_positions[0]

        if response_pos:
            # Next find where "### End" is located.  The model has been trained to end its responses with this
            # sequence (or actually, the token ID it maps to, since it is a special token). We may not find
            # this token, as the response could be truncated.  If we don't find it then just return everything
            # to the end. Note that even though we set eos_token_id, we still see this token at the end.
            end_pos = None
            end_positions = np.where(model_outputs == self.end_key_token_id)[0]
            if len(end_positions) > 0:
                end_pos = end_positions[0]
            decoded = self.tokenizer.decode(model_outputs[response_pos + 1: end_pos]).strip()
        return [decoded]

    def handle(self, data, context):
        """Entry point. It takes the data from the input request and returns
           the predicted outcome for the input.

        Args:
            data (list): The input data that needs to be made a prediction request on.

        Returns:
            list : Returns a list with the predicted responses.
        """
        start_time = time.time()
        data_preprocess = self.preprocess(data)
        output = self.inference(data_preprocess)
        output = self.postprocess(output)
        stop_time = time.time()
        self.context.metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        return output


if __name__ == '__main__':
    import argparse
    from collections import namedtuple
    from ts.metrics.metrics_store import MetricsStore

    Context = namedtuple("Context", ['manifest', 'system_properties', 'metrics'])

    argparser = argparse.ArgumentParser()
    argparser.add_argument('model_dir', type=str)
    argparser.add_argument('prompt', type=str)

    args = argparser.parse_args()

    context = Context(
        manifest={},
        system_properties={
            'model_dir': args.model_dir
        },
        metrics=MetricsStore(request_ids={'a': 'prediction'}, model_name='dolly-v2')
    )

    handler = InstructionTextGenerationHandler()
    handler.initialize(context)
    print(handler.handle([args.prompt], context))
