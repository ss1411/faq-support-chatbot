from openai import OpenAI


class OpenAIClient:
    def __init__(self, api_key: str = None):
        if not api_key:
            raise ValueError('OPENROUTER_API_KEY is required')
        self.api_key = api_key
        try:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",    
                api_key=self.api_key
            )
        except Exception as e:
            raise

    def call_openai(self, user_question: str, context: str, system_msg: str = "You are an intelligent helpful assistant"):
        response = self.client.responses.create(
            model='gpt-4o-mini',
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_question}\n\nAnswer concisely, accurately, and cite chunk ids that support the answer."}
            ],
            temperature=0.0,
            max_output_tokens=200
        )
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        model_response = response.output[0].content[0].text
        return model_response, input_tokens, output_tokens
    
    # The OpenAI GPT-4o mini costs $0.15 per 1 million input tokens and $0.60 per 1 million output tokens
    def calculate_request_cost(self, num_input_tokens: int, num_output_tokens: int) -> float:
        '''Calculate cost based on token count and rates.
        Args:
            num_input_tokens (int): Number of input tokens used in the request.
            num_output_tokens (int): Number of output tokens generated in the response.
        Returns:
            float: Total cost of the request.
        '''
        rate_per_1M_input_tokens = 0.15
        rate_per_1M_output_tokens = 0.60
        input_cost = (num_input_tokens / 1_000_000) * rate_per_1M_input_tokens
        output_cost = (num_output_tokens / 1_000_000) * rate_per_1M_output_tokens
        total_cost = input_cost + output_cost
        return total_cost

