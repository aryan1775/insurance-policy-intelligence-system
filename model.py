import boto3

class Model:
    def __init__(self):
        self.client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )

    def create_response(self,prompt):
        response = self.client.converse(
            modelId="meta.llama3-8b-instruct-v1:0",
            inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
            performanceConfig={"latency": "standard"},
            messages=[{"role": "user", "content": [{"text":prompt}]}]
        )
        answer = response["output"]["message"]["content"][0]["text"]
        return answer

