from argparse import ArgumentParser
from strands import Agent
from strands.models import BedrockModel

model = BedrockModel(
    model_id="apac.amazon.nova-micro-v1:0",
    region_name="ap-northeast-1"
    )

def main():
    parser = ArgumentParser()
    parser.add_argument("message")
    args = parser.parse_args()

    message = args.message
    agent = Agent(model=model)
    agent(message)

if __name__ == "__main__":
    main()