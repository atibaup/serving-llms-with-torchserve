# Serving LLMs with torchserve and Vertex AI


## Installation

1. Create a [hugginface](huggingface.co) account and configure your SSH key
2. install [git lfs](https://git-lfs.com/)
3. clone your favorite LLM (for now this code has been tested only with dolly-v2-3b and gpt2):
```
git lfs install
git clone https://huggingface.co/gpt2 # or git@hf.co:databricks/dolly-v2-3b
cd gpt2
git lfs pull
cd ..
```
4. Create a virtual or conda environment and install `python=3.9.16`,  `pytorch=1.12.1`,  `transformers=4.24` and `accelerate=0.18` (may work with other versions, this was tested
on a Mac M1, running Ventura 13.3.1)
5. If you are trying to use `dolly-v2-3b`, you will need to hack the accelerate library so
that it does the weight offloading when running on CPU devices, by changing
`site-packages/accelerate/big_modeling.py`, line 333:

```
# main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]
# Trying this workaround: https://github.com/huggingface/transformers/issues/18698
main_device = [d for d in device_map.values() if d not in ["disk"]][0]
```
6. set the `GCP_PROJECT` env var (not needed if not deploying to GCP, but you can set it to a dummy variable anyway)
```
export GCP_PROJECT={your_GCP_project_name_or_some_dummy_value}
```

## Usage

If you have set up the installation for `gpt2`, for example:

- Build the docker image `make build-dev APP=gpt2`
- Run the container `make run APP=gpt2` and wait a minute or so for the workers to load
- Send an inference request via curl `make test APP=gpt2` (be patient :D - inference can take minutes on CPU, especially for the dolly models)
- `make help`: see all available targets

```
curl -d '{"instances": ["How to prepare a spanish omelette:"]}' \
                -H "Content-Type: application/json" \
                -X POST http://localhost:7080/predictions/gpt2 | json_pp

{
   "predictions" : [
      "How to prepare a spanish omelette:\n\n1. Prepare a spanish omelette with a spanish cheesecloth.\n\n2. Place the spanish omelette in a large bowl.\n\n3. Add the spanish cheesecloth to the bowl.\n\n4. Cover the bowl with a lid.\n\n5. Place the lid on the spanish omelette.\n\n6. Place the lid on the spanish omelette.\n\n7. Place the lid on the spanish omelette.\n\n8. Place the lid on the spanish omelette.\n\n9. Place the lid on the spanish omelette.\n\n10. Place the lid on the spanish omelette.\n\n11. Place the lid on the spanish omelette.\n\n12. Place the lid on the spanish omelette.\n\n13. Place the lid on the spanish omelette.\n\n14. Place the lid on the spanish omelette.\n\n15. Place the lid on the spanish omelette.\n\n16. Place the lid on the spanish omelette.\n\n17. Place the lid on the spanish omelette.\n\n"
   ]
}
```


# Serving LLMs with torchserve and Vertex AI: Part I

The need for deploying large Machine Learning models (Language or not) is here to stay and
their characteristics (massive memory and computational needs) make their
deployment a bit trickier.

Regardless of how and where we are deploying a model (on-premise or cloud, through
a managed service or other), the common building block to expose a model as a service
are containers. Wrapping models in docker containers allows us to package
all the necessary artifacts into one (more-or-less) portable component that
can be deployed locally, directly on a stand-alone cloud instance or through a
managed ML deployment service such as AWS's Sagemaker or GCP's Vertex AI.

This post is the first of two posts on how to deploy open-source Language Models
in Vertex AI (though many of the tips and the process itself is very similar for other similar
cloud services). In this first post I will talk about packaging PyTorch LLM models using HuggingFace
transformers into a TorchServe docker container. In particular I will provide an example
that works for the recent dolly-v2 models from Databricks and the good-old GPT2 model
(whose 124M parameters make it not that "large" for current standards).
The next post will be about serving that container via Vertex AI.

## torch serving with torchserve

Since here we are interested in PyTorch models, we will use the torchserve library
which wraps PyTorch models around a REST API that allows for:

i. exposing a prediction (inference) endpoint meeting production-grade parallelism and observability needs
ii. exposing a management API to configure existing or new model deployments

To use torch serve with an LLM, we just need four ingredients:

1. Writing a Custom Handler that implements model initialization, preprocessing, inference and post-processing methods
   to handle prediction calls,
2. Running the `torch-model-archiver`, a tool that will package together all the artifacts into a `.mar` file
 containing the model binaries, the tokenizer files and the custom handler,
3. Preparing a configuration file specifying the run-time parameters of our torch-serve deployment
4. Calling the `torchserve` command, passing it the `.mar` file and the configuration file

I won't bore you with the details of how to run each step - check the code in the repository
and the Dockerfile if you want to delve into the nitty gritty.

## Packing things tight

Now that we know how to run inference on our model via torchserve, we want to pack
everything in a docker container, to be deployed as a containerized application
according to our fancy deployment preferences (as a K8S pod, in a single instance, through a managed
ML service, whatever)

Our starting point will be the official [torchserve image](https://hub.docker.com/r/pytorch/torchserve),
which comes in two flavors: cpu or gpu optimized. These images are already quite heavy (2.4Gb and 5.9Gb already)
so we need to pay attention to avoid adding too much additional weight. We will do that through two tricks:

1. The classic "have as few RUN statements as possible" by concatenating commands in as few steps as possible
2. Using a 2-stage build: one first stage to generate the `.mar` artifact and a second stage that only
installs the runtime dependencies and copies the artifact from the previous stage. With this approach we avoid
keeping the intermediate artifacts as layers of the final image, while keeping all the built instructions
within the Dockerfile.

# Serving LLMs with torchserve and Vertex AI: Part II

WIP

## FAQs:

_Q: Why are torchserve base images so large? Can we slim them down?_

I was asking myself the same question and the answer seems to be "no".

For GPUs, the image weight is dominated by three components: the model weights (several Gb depending on specific models), the cuda libraries (1.7Gb) and the torch library (3.2Gb)
which add up to > 5Gb.

For CPUs, same without the CUDA libraries.

```
model-server@4e21d758e215:~$ du -hs model-store/
4.2G    model-store/

model-server@4e21d758e215:/$ cd / & du -h -d 1 usr/local/
...
3.9G    usr/local/lib
...
1.7G    usr/local/cuda-11.7

model-server@4e21d758e215:/$ du -h -d 1 usr/local/lib/python3.8/dist-packages/
...
3.2G    usr/local/lib/python3.8/dist-packages/torch
...
```
I'm quite new to Torch so not sure if there's anyway to slim down the base installation, but `torch` is a [dependency
of torchserve](https://github.com/pytorch/serve/blob/master/requirements/torch_cu102_linux.txt) so not clear that this can be
done.

NOTES

you may need to create a default metadata store for your project:
curl -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json; charset=utf-8" \
    "https://europe-west4-aiplatform.googleapis.com/v1/projects/coherent-racer-379715/locations/europe-west4/metadataStores?metadata_store_id=default"
