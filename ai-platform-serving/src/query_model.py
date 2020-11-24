import argparse
import ast
from googleapiclient import discovery


def query_model(model, project_id, input_records, version=None):

    # Get the AI-Platform prediction service
    service = discovery.build('ml', 'v1')
    model_id = f'projects/{project_id}/models/{model}'

    if version is not None:
        model_id = model_id + f'/versions/{version}'

    response = service.projects().predict(
        name=model_id,
        body={'instances': input_records}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', required=True,
                        help='Name of the AI Platform model')
    parser.add_argument('-p', '--project-id', required=True,
                        help='GCP project id')
    parser.add_argument('-f', '--input-file', required=True,
                        help='File with input instances')
    parser.add_argument('-v', '--version', default=None,
                        help='Version of the model. If not provided, it will '
                             'use the default version')

    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        input_records = f.read().splitlines()

    input_records = [ast.literal_eval(x) for x in input_records]

    query_model(model=args.model,
                project_id=args.project_id,
                input_records=input_records,
                version=args.version)