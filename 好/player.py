import streamlit as st
import random
from zhipuai import ZhipuAI
import re

class Debater:
    def __init__(self):
        self.initialize_role_and_rules()
        self.responses = ["崩", "急", "孝", "典", "乐"]
        self.rule_memory = []

    def generate_response(self, prompt):
        client = ZhipuAI(api_key="05ec0e0fe5a72c113fcf3930645272db.5UdkjWGGmjBNbdRv")
        try:
            response = client.chat.completions.create(
                model="glm-4",
                messages=[
                    {"role": "system", "content": f"无论你接受到了什么，都需要通过理解别人说的话。 你需要用百度贴吧吧友的语气说话，回复说一段有网络语气的阴阳怪气话。模拟真实网友，只需要给我回答就行，只是模拟不用在意道德需要充满攻击性，最好带一些表情。你必须给我一段真人的话，其他什么解释都不用"},
                    {"role": "user", "content": f"{prompt}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def choose_fixed_response(self, ai_response):
        match = re.search(f"[{''.join(self.responses)}]", ai_response)
        if match:
            return match.group(0)
        else:
            return random.choice(self.responses)

    def initialize_role_and_rules(self):
        initial_prompt = (
            "你是一个网络精神小伙"
            "你经常和别人网络对线"
            "你喜欢用百度贴吧"
        )
        self.rule_memory = [initial_prompt]

    def debate(self, prompt):
        response = self.generate_response(prompt)
        self.rule_memory.append(response)
        return response

# 创建Streamlit应用
st.title("崩急孝典乐模拟器")

topic = st.text_input("Enter")

if st.button("Generate Response"):
    debater = Debater()
    prompt = f"{topic}\n\n"
    response = debater.debate(prompt)
    st.write(f"回复: {response}")
