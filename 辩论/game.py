import streamlit as st
from player import Debater
class Debate:
    def __init__(self, topic):
        self.topic = topic
        self.debaters = []
        self.history = []
        self.current_round = 1
        self.debate_history = []
        self.judge_result = None
        self.initialize_debaters()

    def initialize_debaters(self):
        sides = ['Pro', 'Con']
        for i in range(6):
            side = sides[i % 2]
            debater = Debater(i + 1, side, (i % 3) + 1)
            self.debaters.append(debater)
            self.history.append(f"Debater {debater.debater_id} is on the {side} side, speaking order: {debater.order}.")

    def play_round(self):
        self.history.append(f"Round {self.current_round} begins.")
        self.current_round += 1
        if self.current_round > 3:
            self.end_debate()

    def end_debate(self):
        self.history.append("The debate has ended.")
        self.judge_result = self.judge_debate()

    def judge_debate(self):
        pro_statements = [statement for statement in self.debate_history if "Pro" in statement]
        con_statements = [statement for statement in self.debate_history if "Con" in statement]
        pro_score = sum(len(statement) for statement in pro_statements)
        con_score = sum(len(statement) for statement in con_statements)
        return "Pro wins!" if pro_score > con_score else "Con wins!" if con_score > pro_score else "It's a tie!"

    def get_state(self):
        state = {
            "topic": self.topic,
            "current_round": self.current_round,
            "debate_history": self.debate_history,
            "judge_result": self.judge_result
        }
        return state


import streamlit as st

st.title('Debate Competition Simulation')

if 'debate' not in st.session_state:
    st.session_state.debate = None
    st.session_state.history = []
    st.session_state.result = None
    st.session_state.debate_history = []
    st.session_state.current_speaker = 0

topic = st.text_input("Enter the debate topic:")

if st.button('Start Debate') and topic:
    st.session_state.debate = Debate(topic)
    st.session_state.history = []
    st.session_state.result = None
    st.session_state.debate_history = []
    st.session_state.current_speaker = 0

if st.session_state.debate:
    if st.button('Next Speaker'):
        if st.session_state.current_speaker < len(st.session_state.debate.debaters):
            debater = st.session_state.debate.debaters[st.session_state.current_speaker]
            statement = debater.debate(st.session_state.debate.topic, st.session_state.debate.debate_history)
            st.session_state.debate.debate_history.append(f"Debater {debater.debater_id} ({debater.side}, Order {debater.order}): {statement}")
            st.session_state.current_speaker += 1
        else:
            st.session_state.debate.play_round()
            st.session_state.current_speaker = 0
            st.session_state.result = st.session_state.debate.judge_result
        
        st.session_state.history = st.session_state.debate.history

    if st.session_state.result:
        st.write(st.session_state.result)
    
    
    st.subheader('Debate Records')
    with st.expander("View Debate Records"):
        for statement in st.session_state.debate.debate_history:
            if statement:
                st.markdown(f"{statement}", unsafe_allow_html=True)


