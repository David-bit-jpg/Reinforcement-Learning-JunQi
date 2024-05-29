import random
import streamlit as st
from player import Player

class AvalonGame:
    def __init__(self, num_players=7):
        self.num_players = num_players
        self.players = []
        self.roles = ['Merlin', 'Percival', 'Loyal Servant', 'Loyal Servant', 'Assassin', 'Mordred', 'Morgana']
        random.shuffle(self.roles)
        self.history = []
        self.current_quest = 1
        self.quest_results = []
        self.debate_history = []
        self.inner_monologue_history = []
        self.experience_pool = []
        self.assign_roles()

    def assign_roles(self):
        for i in range(self.num_players):
            player = Player(i + 1, self.roles[i])
            self.players.append(player)
            self.history.append(f"Player {player.player_id} is assigned the role {player.role}")

        evil_players = [player for player in self.players if player.role in ['Assassin', 'Mordred', 'Morgana']]
        for player in self.players:
            if player.role in ['Assassin', 'Mordred', 'Morgana']:
                player.set_known_evil([ep.player_id for ep in evil_players if ep.player_id != player.player_id])

        try:
            merlin = next(player for player in self.players if player.role == 'Merlin')
            merlin.set_known_evil([ep.player_id for ep in evil_players if ep.role != 'Mordred'])
        except StopIteration:
            pass

        try:
            percival = next(player for player in self.players if player.role == 'Percival')
            percival.set_merlin_known_by([p.player_id for p in self.players if p.role in ['Merlin', 'Morgana']])
        except StopIteration:
            pass

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
        debate_feedback = [leader_statement]
        for other_player in self.players:
            if other_player.player_id != leader.player_id:
                statement = other_player.debate(proposed_team)
                self.debate_history.append(f"Player {other_player.player_id}: {statement}")
                debate_feedback.append(statement)
                for player in self.players:
                    player.remember(f"Player {other_player.player_id}: {statement}")

        # Leader comfirm final team choice
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

    def vote_inner_monologue(self, team):
        self.inner_monologue_history.append(f"<b>Inner Monologue for round {self.current_quest}:</b>")
        for player in self.players:
            monologue = player.vote_for_team_prompt(team)
            self.inner_monologue_history.append(f"Player {player.player_id}: {monologue}")

    def execute_quest(self, team):
        quest_result = [self.players[player_id - 1].execute_quest() for player_id in team]
        self.history.append(f"Team {team} executes quest with results: {quest_result.count('Success')} Success, {quest_result.count('Fail')} Fail")
        for player in self.players:
            player.remember(f"Team {team} executes quest with results: {quest_result.count('Success')} Success, {quest_result.count('Fail')} Fail")
        return quest_result

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
        self.vote_inner_monologue(final_team)
        votes = self.vote_for_team(leader, final_team)
        if list(votes.values()).count('Approve') > self.num_players / 2:
            quest_result = self.execute_quest(final_team)
            self.quest_results.append(quest_result)
        else:
            self.history.append(f"Team {final_team} was not approved.")
        self.current_quest += 1

        for player in self.players:
            reflection = player.reflect()
            self.experience_pool.append(reflection)

        return f"Round {self.current_quest - 1} complete"

# Streamlit 界面代码
st.title('Avalon Game Simulation')

if 'game' not in st.session_state:
    st.session_state.game = AvalonGame()
    st.session_state.history = []
    st.session_state.result = None
    st.session_state.debate_history = []
    st.session_state.inner_monologue_history = []

if st.button('Start New Game'):
    st.session_state.game = AvalonGame()
    st.session_state.history = []
    st.session_state.result = None
    st.session_state.debate_history = []
    st.session_state.inner_monologue_history = []

if st.button('Next Round'):
    result = st.session_state.game.play_round()
    if result == "Game over":
        st.session_state.result = result
    st.session_state.history = st.session_state.game.history
    st.session_state.debate_history = st.session_state.game.debate_history
    st.session_state.inner_monologue_history = st.session_state.game.inner_monologue_history

if st.session_state.result:
    st.write(st.session_state.result)

st.subheader('Debate Records')
with st.expander("View Debate Records"):
    for statement in st.session_state.debate_history:
        if statement:
            st.markdown(statement, unsafe_allow_html=True)
        else:
            st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Voting Inner Monologue Records')
with st.expander("View Inner Monologue Records"):
    for monologue in st.session_state.inner_monologue_history:
        if monologue:
            st.markdown(monologue, unsafe_allow_html=True)
        else:
            st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Game Records')
for record in st.session_state.history:
    if record:
        st.markdown(record, unsafe_allow_html=True)
    else:
        st.markdown("<br>", unsafe_allow_html=True)
