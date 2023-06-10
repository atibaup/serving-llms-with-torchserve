from transformers import GPT2Tokenizer, GPT2LMHeadModel
from ts.torch_handler.base_handler import BaseHandler
import torch

import logging
from typing import List, Union, Dict
import time

logger = logging.getLogger(__name__)


class GPT2GenerationHandler(BaseHandler):
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

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(
            model_dir,
            device_map="auto",
            offload_folder="offload",
            offload_state_dict=True
        )
        self.model.eval()
        logger.debug('GPT2 model from path {} loaded successfully'.format(model_dir))

        self.initialized = True

    def preprocess(self, text: List[str]) -> Dict[str, Union[str, torch.Tensor]]:
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing

        Args :
            text (list): List of the data from the request input.

        Returns:
            tensor: Returns the tensor data of the input
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs

    def inference(self, model_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        generated_sequence = self.model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k
        ).cpu()
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
        #return self.tokenizer.decode(model_outputs)
        return self.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)

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
        metrics=MetricsStore(request_ids={'a': 'prediction'}, model_name='gpt2')
    )

    handler = GPT2GenerationHandler()
    handler.initialize(context)
    print(handler.handle([args.prompt], context))
