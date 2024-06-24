import random
from sentence_transformers import SentenceTransformer, util
import torch
from zhipuai import ZhipuAI
import re

class Debater:
    def __init__(self, debater_id, side, order):
        self.debater_id = debater_id
        self.side = side
        self.order = order
        self.released_statement = []
        self.initialize_role_and_rules()

    def generate_response(self, prompt):
        client = ZhipuAI(api_key="05ec0e0fe5a72c113fcf3930645272db.5UdkjWGGmjBNbdRv")
        try:
            response = client.chat.completions.create(
                model="glm-4",
                messages=[
                    {"role": "system", "content": f"You are a debater in a debate competition. You are Debater {self.debater_id}, and your side is {self.side}. Your position in the debate order is {self.order}."},
                    {"role": "user", "content": f"Consider the prompt: {prompt}. Give a response like a real human debater in conversation, focusing on the spoken aspects and not on internal thoughts."}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def initialize_role_and_rules(self):
        initial_prompt = (
            "You are participating in a debate competition. Each side has three debaters. "
            "Your goal is to present strong arguments to support your side and counter the opposing side's arguments."
        )
        self.rule_memory = [initial_prompt]

    def debate_statement_prompt(self, topic, history):
        base_prompt = (
            f"{self.rule_memory}"
            f"You are Debater {self.debater_id} with side {self.side}. "
            f"Current topic: {topic}. Your position in the debate order is {self.order}. "
            f"Review the debate history and focus on rebutting the previous debater's points while reinforcing your own side's key arguments.\n"
            f"Debate history:\n{history}\n"
            f"Generate a strong argument that addresses the previous debater's points and provides evidence, examples, and logical reasoning to support your side. "
            f"Respond in a convincing and engaging manner."
        )
        return base_prompt

    def debate(self, topic, history):
        prompt = self.debate_statement_prompt(topic, history)
        response = self.generate_response(prompt)
        self.released_statement.append(response)
        return response
