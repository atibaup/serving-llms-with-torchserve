import os
from collections import namedtuple
from typing import Tuple

from google.cloud import aiplatform
from google.oauth2 import service_account
import yaml

DeploymentConfig = namedtuple(
    "DeploymentConfig",
    ["machine_type", "min_replica_count", "max_replica_count",
     "accelerator_type", "accelerator_count", "service_account", "traffic_percentage"]
)

AiPlatformConfig = namedtuple(
    "AiPlatformConfig",
    ["sa_credentials", "gcp_staging_bucket"]
)


GCP_PROJECT = os.environ['GCP_PROJECT']
GCP_REGION = os.environ['GCP_REGION']


def create_vertexai_model(model_name: str, container_image_uri: str, **kwargs) -> aiplatform.Model:
    health_route = "/ping"
    predict_route = f"/predictions/{model_name}"
    serving_container_ports = [7080]

    models = aiplatform.Model.list(filter=f'display_name="{model_name}"')

    if models:
        assert len(models) == 1, f"Error: More than one model named {model_name}"
        model = models[0]
    else:
        model = aiplatform.Model.upload(
            display_name=model_name,
            serving_container_image_uri=container_image_uri,
            serving_container_predict_route=predict_route,
            serving_container_health_route=health_route,
            serving_container_ports=serving_container_ports,
            **kwargs
        )
        print(f"Creating {model_name} model...")

    model.wait()
    print(f"Model created:\n- model.display_name: {model.display_name}\n- model.resource_name:{model.resource_name}")
    return model


def deploy_model_to_endpoint(model: aiplatform.Model, deployment_config: DeploymentConfig) -> Tuple[aiplatform.Endpoint, aiplatform.Model]:
    deployed_model_display_name = model.display_name
    endpoint_display_name = f"{deployed_model_display_name}-endpoint"

    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_display_name}"')

    if endpoints:
        assert len(endpoints) == 1
        endpoint = endpoints[0]
        print(f"Using existing endpoint: {endpoint.display_name}")
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)

    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_display_name,
        sync=True,
        **deployment_config._asdict(),
    )

    print("Waiting for model deployment...")
    model.wait()
    print(f"Model deployed at: {endpoint.name}")
    return endpoint, model


def load_config(path: str) -> Tuple[DeploymentConfig, AiPlatformConfig]:
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    deployment_config = DeploymentConfig(**config['deployment'])
    aiplatform_config = AiPlatformConfig(**config['aiplatform'])
    return deployment_config, aiplatform_config


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("app")
    arg_parser.add_argument("version")
    arg_parser.add_argument("container_image_name")
    arg_parser.add_argument("--config", default='config.yml')

    args = arg_parser.parse_args()

    deployment_config, aiplatform_config = load_config(args.config)
    credentials = service_account.Credentials.from_service_account_file(aiplatform_config.sa_credentials)

    aiplatform.init(
        project=GCP_PROJECT,
        location=GCP_REGION,
        staging_bucket=aiplatform_config.gcp_staging_bucket,
        credentials=credentials,
        experiment=f'{args.app}-experiments'
    )

    model_name = f"{args.app}-v{args.version}"
    container_image_uri = f"gcr.io/{GCP_PROJECT}/{args.container_image_name}"
    model = create_vertexai_model(model_name, container_image_uri)
    deploy_model_to_endpoint(model, deployment_config)
