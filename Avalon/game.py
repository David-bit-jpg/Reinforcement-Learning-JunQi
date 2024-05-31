import random
import streamlit as st
from player import Player
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer, util

class ExperiencePool:
    def __init__(self):
        self.pool = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 使用 Sentence-BERT 模型

    def add_experience(self, player_id, role, reflection, response, score):
        self.pool.append((player_id, role, reflection, response, score))

    def get_suggestions(self, current_reflection):
        relevant_experiences = [exp for exp in self.pool if self.is_relevant(exp, current_reflection)]
        suggestions = self.extract_suggestions(relevant_experiences)
        return suggestions

    def is_relevant(self, experience, current_reflection):
        # 使用 Sentence-BERT 模型计算相似度，判断经验是否与当前反思相关
        reflection_embedding = self.model.encode(current_reflection, convert_to_tensor=True)
        experience_embedding = self.model.encode(experience[2], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(reflection_embedding, experience_embedding)
        threshold = 0.8  # 设置相似度阈值
        return similarity > threshold

    def extract_suggestions(self, experiences):
        # 从相关经验中提取建议的逻辑
        if not experiences:
            return "No relevant experiences found."

        # 获取评分中位数的经验
        scores = [exp[4] for exp in experiences]
        median_score = sorted(scores)[len(scores) // 2]
        median_experiences = [exp for exp in experiences if exp[4] == median_score]

        # 获取得分最低的经验
        min_score = min(scores)
        min_experience = next(exp for exp in experiences if exp[4] == min_score)

        # 提取建议
        suggestions = []
        for exp in median_experiences:
            suggestions.append(f"From experience of Player {exp[0]} (Role: {exp[1]}): {exp[3]}")

        suggestions.append(f"Learn from this mistake by Player {min_experience[0]} (Role: {min_experience[1]}): {min_experience[3]}")

        return suggestions
class AvalonGame:
    def __init__(self, num_players=7):
        self.num_players = num_players
        self.players = []
        self.roles = ['Merlin', 'Percival', 'Loyal Servant', 'Loyal Servant', 'Assassin', 'Oberon', 'Morgana']
        random.shuffle(self.roles)
        self.history = []
        self.current_quest = 1
        self.quest_results = []
        self.debate_history = []
        self.inner_monologue_history = []
        self.experience_pool = ExperiencePool()
        self.game_count = 0
        self.assign_roles()

    def assign_roles(self):
        for i in range(self.num_players):
            player = Player(i + 1, self.roles[i], self.experience_pool)
            self.players.append(player)
            self.history.append(f"Player {player.player_id} is assigned the role {player.role}")

        evil_players = [player for player in self.players if player.role in ['Assassin', 'Oberon', 'Morgana']]
        good_players = [player for player in self.players if player.role in ['Merlin', 'Percival', 'Loyal Servant']]
        
        # Set known evil players for Merlin (including Oberon)
        try:
            merlin = next(player for player in self.players if player.role == 'Merlin')
            merlin_known_evil = [ep.player_id for ep in evil_players]
            merlin.set_known_evil(merlin_known_evil)
            merlin.rule_memory.append(f"As Merlin, you know the evil players are: {merlin_known_evil}. Try to hide your identity and gain trust to good players.")
        except StopIteration:
            pass

        # Set known evil players for evil roles (excluding Oberon)
        for player in self.players:
            if player.role in ['Assassin', 'Morgana']:
                known_evil = [ep.player_id for ep in evil_players if ep.role != 'Oberon' and ep.player_id != player.player_id]
                player.set_known_evil(known_evil)
                player.rule_memory.append(f"As {player.role}, you know the other evil players are: {known_evil}.")

        # Oberon does not know other evil players
        try:
            oberon = next(player for player in self.players if player.role == 'Oberon')
            oberon.set_known_evil([])
            oberon.rule_memory.append("As Oberon, you do not know the other evil players. Try to identify who are evil and gain trust to join them.")
        except StopIteration:
            pass

        # Set known Merlin/Morgana for Percival
        try:
            percival = next(player for player in self.players if player.role == 'Percival')
            percival_known = [p.player_id for p in self.players if p.role in ['Merlin', 'Morgana']]
            percival.set_merlin_known_by(percival_known)
            percival.rule_memory.append(f"As Percival, you know Merlin and Morgana are: {percival_known}. You need to figure out who is who.")
        except StopIteration:
            pass

    def reset_game(self):
        self.history = []
        self.current_quest = 1
        self.quest_results = []
        self.debate_history = []
        self.inner_monologue_history = []
        self.players = []
        self.assign_roles()
        self.game_count += 1

    def end_game(self, result):
        if result == "good_win":
            self.history.append("Good team has won the quests.")
            assassin = next((player for player in self.players if player.role == "Assassin"), None)
            if assassin:
                self.history.append("Assassin gets a chance to assassinate Merlin.")
                merlin_guess = assassin.assassinate_merlin_prompt()
                merlin = next((player for player in self.players if player.role == "Merlin"), None)
                if merlin and merlin_guess.strip() == f"Player {merlin.player_id}":
                    self.history.append("Assassin correctly identified Merlin. Evil team wins!")
                    return "Evil wins!"
                else:
                    self.history.append("Assassin failed to identify Merlin. Good team wins!")
                    return "Good wins!"
        else:
            self.history.append("Evil team has won.")
            return "Evil wins!"

    def play_round(self):
        if self.current_quest > 5:
            return self.end_game("good_win" if sum(1 for result in self.quest_results if result.count('Fail') == 0) >= 3 else "evil_win")

        self.history.append("")
        leader = self.players[(self.current_quest - 1) % self.num_players]
        team_size = 2 if self.current_quest in [1, 2] else 3
        initial_team = self.nominate_team(leader, team_size)
        final_team = self.debate(leader, initial_team)
        votes = self.vote_for_team(leader, final_team)
        if list(votes.values()).count('Approve') > self.num_players / 2:
            quest_result = self.execute_quest(final_team)
            self.quest_results.append(quest_result)
        else:
            self.history.append(f"Team {final_team} was not approved.")
        self.current_quest += 1

        for player in self.players:
            reflection = player.reflect()
            self.experience_pool.add_experience(player.player_id, player.role, reflection, "", 0)

        return f"Round {self.current_quest - 1} complete"

    def nominate_team(self, leader, team_size):
        team = leader.nominate_team(team_size)
        self.history.append(f"<b>Leader {leader.player_id} nominates team: {team}</b>")
        for player in self.players:
            player.remember(f"Leader {leader.player_id} nominates team: {team}")
        return team

    def debate(self, leader, proposed_team):
        self.debate_history.append(f"<b>Debate for round {self.current_quest}:</b>")
        self.debate_history.append(f"Player {leader.player_id} (Leader): I propose the team: {', '.join(['Player ' + str(player) for player in proposed_team])}.")
        
        # leader statement
        leader_statement = leader.debate(proposed_team, is_leader=True)
        self.debate_history.append(f"Player {leader.player_id} (Leader): {leader_statement}")
        
        # other players
        debate_feedback = [f"Player {leader.player_id} (Leader): {leader_statement}"]
        for other_player in self.players:
            if other_player.player_id != leader.player_id:
                statement = other_player.debate(proposed_team, previous_statements=debate_feedback)
                self.debate_history.append(f"Player {other_player.player_id}: {statement}")
                debate_feedback.append(f"Player {other_player.player_id}: {statement}")
                for player in self.players:
                    player.remember(f"Player {other_player.player_id}: {statement}")

        # Leader confirm final team choice
        final_team = leader.finalize_team(proposed_team, debate_feedback)
        self.history.append(f"<b>Leader {leader.player_id} finalizes the team: {final_team}</b>")
        for player in self.players:
            player.remember(f"Leader {leader.player_id} finalizes the team: {final_team}")
        return final_team

    def vote_for_team(self, leader, team):
        votes = {player.player_id: player.vote_for_team(team) if player.player_id != leader.player_id else 'Approve' for player in self.players}
        self.history.append(f"Team {team} voting results: {votes}")
        for player in self.players:
            player.remember(f"Team {team} voting results: {votes}")
        return votes

    def execute_quest(self, team):
        quest_result = [self.players[player_id - 1].execute_quest() for player_id in team]
        self.history.append(f"Team {team} executes quest with results: {quest_result.count('Success')} Success, {quest_result.count('Fail')} Fail")
        for player in self.players:
            player.remember(f"Team {team} executes quest with results: {quest_result.count('Success')} Success, {quest_result.count('Fail')} Fail")
        return quest_result

# Streamlit 界面代码
st.title('Avalon Game Simulation')

if 'game' not in st.session_state:
    st.session_state.game = AvalonGame()
    st.session_state.history = []
    st.session_state.result = None
    st.session_state.debate_history = []
    st.session_state.game_count = 0

if st.button('Start New Game'):
    st.session_state.game.reset_game()
    st.session_state.history = []
    st.session_state.result = None
    st.session_state.debate_history = []
    st.session_state.game_count += 1

if st.button('Next Round'):
    result = st.session_state.game.play_round()
    if result == "Game over":
        st.session_state.result = result
    st.session_state.history = st.session_state.game.history
    st.session_state.debate_history = st.session_state.game.debate_history

if st.session_state.result:
    st.write(st.session_state.result)

st.subheader('Debate Records')
with st.expander("View Debate Records"):
    for statement in st.session_state.debate_history:
        if statement:
            st.markdown(statement, unsafe_allow_html=True)
        else:
            st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Game Records')
for record in st.session_state.history:
    if record:
        st.markdown(record, unsafe_allow_html=True)
    else:
        st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Game Count')
st.write(f"Games played: {st.session_state.game_count}")
