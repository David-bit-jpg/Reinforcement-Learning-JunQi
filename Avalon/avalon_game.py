import random
import streamlit as st
import numpy as np
from player import Player

class Avalon:
    def __init__(self, num_players=7):
        self.num_players = num_players
        self.players = []
        self.roles = ['Merlin', 'Percival', 'Loyal Servant', 'Loyal Servant', 'Assassin', 'Oberon', 'Morgana']
        random.shuffle(self.roles)
        self.history = []
        self.current_quest = 1
        self.quest_results = []
        self.quest_history = []
        self.current_score = {'Good': 0, 'Evil': 0}
        self.debate_history = []
        self.game_count = 0
        self.assign_roles()
        self.current_leader = 0
        self.current_leader_attempts = 0

    def assign_roles(self):
        for i in range(self.num_players):
            player = Player(i + 1, self.roles[i])
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

    def get_state(self):
        state = {
            "roles": [player.role for player in self.players],
            "current_quest": self.current_quest,
            "quest_history": self.quest_history,
            "votes": [player.vote_for_team for player in self.players]
        }
        return state

    def step(self, action):
        reward = 0
        done = False

        if action == "propose_team":
            pass
        elif action == "vote_for_team":
            pass
        elif action == "execute_quest":
            pass

        next_state = self.get_state()
        return next_state, reward, done

    def available_actions(self):
        actions = ["propose_team", "vote_for_team", "execute_quest"]
        return actions

    def reset_game(self):
        self.history = []
        self.current_quest = 1
        self.quest_results = []
        self.quest_history = [] 
        self.debate_history = []
        self.players = []
        self.assign_roles()
        self.current_leader = 0 
        self.current_score = {'Good': 0, 'Evil': 0}
        self.current_leader_attempts = 0 
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
        leader = self.players[self.current_leader]

        if self.current_quest in [1]:
            team_size = 2 
        elif self.current_quest in [2,3]:
            team_size = 3 
        else:
            team_size = 4 
        initial_team = self.nominate_team(leader, team_size, self.current_score)
        final_team = self.debate(leader, initial_team, team_size, self.current_score)
        votes = self.vote_for_team(leader, final_team, self.current_score)
        self.current_leader = (self.current_leader + 1) % self.num_players
        if list(votes.values()).count('Approve') > self.num_players / 2:
            quest_result = self.execute_quest(final_team, self.current_score)
            self.quest_results.append(quest_result)
            self.quest_history.append(f"Quest {self.current_quest} with Team member {final_team} {'Failed' if quest_result.count('Fail') > 0 else 'Succeeded'}")
            for player in self.players:
                player.remember_quest(f"Quest {self.current_quest} with Team member {final_team} {'Failed' if quest_result.count('Fail') > 0 else 'Succeeded'}")
            self.current_quest += 1
            self.current_leader_attempts = 0
        else:
            self.history.append(f"Team {final_team} was not approved.")
            self.current_leader_attempts += 1
            if self.current_leader_attempts >= 3:
                self.history.append("Forcing the team to go on the quest after three failed attempts.")
                quest_result = self.execute_quest(final_team, self.current_score)
                self.quest_results.append(quest_result)
                self.quest_history.append(f"Quest {self.current_quest} with Team member {final_team} {'Failed' if quest_result.count('Fail') > 0 else 'Succeeded'}")
                for player in self.players:
                    player.remember_quest(f"Quest {self.current_quest} with Team member {final_team} {'Failed' if quest_result.count('Fail') > 0 else 'Succeeded'}")
                self.current_quest += 1
                self.current_leader_attempts = 0
        for player in self.players:
            player.update_core_memory_with_inferences(" ".join(player.short_term_memory))
        return f"Round {self.current_quest - 1} complete"

    def nominate_team(self, leader, team_size, current_score):
        team_response = leader.nominate_team(team_size, current_score)
        self.debate_history.append(f"<b>Leader  {leader.player_id} (Leader)</b>: {team_response}")
        for player in self.players:
            player.remember(f"Player {leader.player_id} (Leader): {team_response}")
        team = leader.parse_team_from_response(team_response, team_size)
        self.history.append(f"<b>Leader {leader.player_id} nominates team: {team}</b>")
        for player in self.players:
            player.remember(f"Player {leader.player_id} nominates team: {team}")
        return team
    
    def debate(self, leader, proposed_team, team_size, current_score):
        self.debate_history.append(f"<b>Debate for round {self.current_quest}:</b>")
        self.debate_history.append(f"Player {leader.player_id} (Leader): I propose the team: {', '.join(['Player ' + str(player) for player in proposed_team])}.")
        for player in self.players:
            player.update_core_memory_with_inferences(" ".join(player.short_term_memory))
        # Leader's initial statement
        leader_statement = leader.debate(proposed_team, is_leader=True, current_score=current_score)
        self.debate_history.append(f"Player {leader.player_id} (Leader): {leader_statement}")
        debate_feedback = [f"Player {leader.player_id} (Leader): {leader_statement}"]
        for player in self.players:
            player.remember(f"Player {leader.player_id} (Leader): {leader_statement}")
        
        # Other players' statements
        for other_player in self.players:
            if other_player.player_id != leader.player_id:
                statement = other_player.debate(proposed_team, previous_statements=debate_feedback, current_score=current_score)
                self.debate_history.append(f"Player {other_player.player_id}: {statement}")
                debate_feedback.append(f"Player {other_player.player_id}: {statement}")
                for player in self.players:
                    player.remember(f"Player {other_player.player_id}: {statement}")

        # Leader finalizes the team after debate
        final_team_response = leader.finalize_team(proposed_team, debate_feedback, current_score)
        self.debate_history.append(f"<b>Leader {leader.player_id}</b>: {final_team_response}")
        debate_feedback.append(f"Player {leader.player_id}: {final_team_response}")
        for player in self.players:
            player.remember(f"Player {leader.player_id}: {final_team_response}")
            
        final_team = leader.parse_team_from_response(final_team_response, team_size)
        self.history.append(f"<b>Leader {leader.player_id} finalizes the team: {final_team}</b>")
        for player in self.players:
            player.remember(f"Player {leader.player_id} finalizes the team: {final_team}")
        return final_team

    def vote_for_team(self, leader, team, current_score):
        votes = {player.player_id: player.vote_for_team(team, current_score) for player in self.players}
        self.history.append(f"Team {team} voting results: {votes}")
        for player in self.players:
            player.remember(f"Team {team} voting results: {votes}")

        if list(votes.values()).count('Approve') > self.num_players / 2:
            return votes
        else:
            self.history.append(f"Team {team} was not approved by the majority.")
            return votes

    def execute_quest(self, team, current_score):
        quest_result = [self.players[player_id - 1].execute_quest() for player_id in team]
        success_count = quest_result.count('Success')
        fail_count = quest_result.count('Fail')

        if fail_count > 0:
            current_score['Evil'] += 1
        else:
            current_score['Good'] += 1

        self.history.append(f"Team {team} executes quest with results: {success_count} Success, {fail_count} Fail")
        for player in self.players:
            player.remember(f"Team {team} executes quest with results: {success_count} Success, {fail_count} Fail")

        return quest_result


# Streamlit 界面代码
st.title('Avalon Game Simulation')

if 'game' not in st.session_state:
    st.session_state.game = Avalon()
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

st.subheader('Quest History')
for quest_record in st.session_state.game.quest_history:
    st.write(quest_record)

st.subheader('Game Count')
st.write(f"Games played: {st.session_state.game_count}")
