import random
import openai

openai.api_key = 'sk-proj-Nav5sY3PPWTa4F8fCvM7T3BlbkFJJfpnDZWH8sTAplVnwtry'

class Player:
    def __init__(self, player_id, role):
        self.player_id = player_id
        self.role = role
        self.short_term_memory = []
        self.long_term_memory = []
        self.known_evil = []
        self.is_merlin_known_by = []
        self.initialize_role_and_rules()

    def set_known_evil(self, known_evil):
        self.known_evil = known_evil

    def set_merlin_known_by(self, merlin_known_by):
        self.is_merlin_known_by = merlin_known_by

    def remember(self, info):
        self.short_term_memory.append(info)
        self.long_term_memory.append(info)

    def filter_memory(self):
        filtered_memory = []
        for memory in self.long_term_memory:
            if any(keyword in memory for keyword in ["evil", "good", "Merlin", "Morgana", "Assassin", "Percival", "Mordred", "role", "success", "fail", "approve", "reject"]):
                filtered_memory.append(memory)
        return filtered_memory[-50:]

    def initialize_role_and_rules(self):
        game_rule_prompt = (
            "You are playing a seven-person Avalon game like a real human. Each role has unique challenges and responsibilities.\n"
            "Introduction to Roles in Avalon Game:\n"
            "Merlin knows evil players but must remain subtle.\n"
            "Percival must discern the real Merlin from Morgana.\n"
            "Loyal Servant of Arthur relies on voting and discussion.\n"
            "Morgana impersonates Merlin to deceive Percival.\n"
            "Assassin seeks Merlin for a final assassination if good wins.\n"
            "Mordred is evil but hidden from Merlin.\n"
            "Roles: One Merlin, one Percival, two Loyal Servant of Arthur, one Morgana, one Assassin, one Mordred.\n"
            "Objective: Lead your team to victory with limited information."
        )

        game_role_hint_prompt = (
            "You are playing a seven-person Avalon game. Here are your role hints:\n"
            "Merlin:\n"
            "- You are the leader of good players and must not expose your identity.\n"
            "- Know the identities of evil players.\n"
            "- Subtly guide your team, especially Percival.\n"
            "- Avoid behaviors that expose your role: overly accusing, being too helpful.\n"
            "- Goal: Win without revealing identity.\n"
            "Percival:\n"
            "- You are on the good side and want to win against the evil side.\n"
            "- Know identities of Merlin and Morgana, but unsure who is who.\n"
            "- Use subtle hints to guide the team and protect Merlin.\n"
            "- Be cautious not to expose Merlin while deciphering true identities.\n"
            "- Goal: Win while safeguarding Merlin.\n"
            "Loyal Servant of Arthur:\n"
            "- You are on the good side and want to win against the evil side.\n"
            "- No special knowledge, rely on discussion and voting.\n"
            "- Contribute to the success of Quests.\n"
            "- Goal: Win by helping complete Quests and protecting Merlin.\n"
            "Morgana:\n"
            "- You are on the evil side and want to win against the good side.\n"
            "- Pretend to be Merlin to mislead Percival and the good side.\n"
            "- Work to prevent Quests' success.\n"
            "- Goal: Confuse and sabotage to win.\n"
            "Assassin:\n"
            "- You are on the evil side and want to win against the good side.\n"
            "- Discreetly seek Merlin's identity.\n"
            "- Work to prevent Quests' success.\n"
            "- When confident who Merlin is, assassinate.\n"
            "- Goal: Win by Quest failures or assassinating Merlin.\n"
            "Mordred:\n"
            "- You are on the evil side and want to win against the good side.\n"
            "- You are evil, but hidden from Merlin.\n"
            "- Work with other evil players to sabotage Quests.\n"
            "- Goal: Win by causing Quest failures."
        )

        initial_prompt = (
            f"{game_rule_prompt}\n\n"
            f"{game_role_hint_prompt}\n\n"
            f"You are Player {self.player_id}, your role is {self.role}."
        )

        openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player learning the game rules and your role."},
                {"role": "user", "content": initial_prompt}
            ]
        )

    def reflect(self):
        first_order_prompt = (
            f"You are Player {self.player_id} with role {self.role}. "
            f"Current situation: Reflect on the current game state and guess the roles of other players. "
            f"Analyze other players based on game dialogues with roles: Merlin, Percival, Loyal Servant of Arthur, Morgana, Assassin, Mordred. Morgana, Assassin, and Mordred are evil; others are good.\n"
            f"Consider:\n"
            f"1. Quest Outcomes: Take into account the results of past missions to analyze players' roles.\n"
            f"2. Role List: Remember the possible roles in the game—Merlin, Percival, two Loyal Servants, Morgana, Assassin, and Mordred—and their alignments.\n"
            f"3. Level of Certainty: Use 'Certain' or 'Unknown' to gauge your confidence in your role guesses for each player.\n"
            f"4. Players Disclosing Evil Roles: Be cautious around players who have openly claimed or hinted at being evil roles like Morgana or Assassin.\n"
            f"5. Prior Guesses: Reflect on your earlier estimations of other players' roles, but don't rely solely on them."
        )
        first_order_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player reflecting on the current game state."},
                {"role": "user", "content": first_order_prompt}
            ]
        )
        initial_reflection = first_order_response.choices[0].message['content']

        formulation_prompt = (
            f"Respond in two stages: THINK and SPEAK\n"
            f"In think, internally strategize using history and consider possible deception.\n"
            f"In speak, organize your language based on your contemplation and speak accordingly.\n"
            f"Understand your role's main objective and break it down into chronological sub-goals based on game history. Your thought process should follow these sub-goals for a systematic approach to the main goal.\n"
            f"\nYou are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on the current situation based on the following memories: {self.short_term_memory}. Summarize the key points and your analysis."
        )
        formulation_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player reflecting on the current game state."},
                {"role": "user", "content": formulation_prompt}
            ]
        )
        initial_reflection = formulation_response.choices[0].message['content']

        perspective_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on your initial thoughts: {initial_reflection}. "
            f"Analyze how your original SPEAK content might be interpreted by other game roles. Reflect on whether it may inadvertently reveal your role-specific clues.\n"
            f"Consider:\n"
            f"1. The perspectives of each game role, including their probable reactions to your SPEAK content.\n"
            f"2. Any unique hints or clues in your original SPEAK that might disclose your role."
        )
        perspective_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player reflecting on how others might perceive your thoughts."},
                {"role": "user", "content": perspective_prompt}
            ]
        )
        other_players_perspective = perspective_response.choices[0].message['content']

        refinement_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on your initial thoughts and others' perspective: {other_players_perspective}. "
            f"Now refine your thoughts to ensure your role and intentions remain concealed or properly presented.\n"
            f"\nYou're observing Player {self.player_id} with role {self.role}. Current situation: {self.short_term_memory}.\n"
            f"\nYour task is to:\n"
            f"1. Evaluate if Player {self.player_id}'s actions align with {self.role}.\n"
            f"2. Improve Player {self.player_id}'s chances of winning through your previous second perspective transition thought.\n"
            f"3. Keep role hint in public dialogue.\n"
            f"Consider:\n"
            f"1. Target Outcome: Aim to achieve the desired result as your role dictates in the game.\n"
            f"2. Role Alignment: Evaluate whether your THINK and SPEAK contents align well with your role {self.role} in the current game state.\n"
            f"3. Strategy Reevaluation: Consider what changes could be made to your THINK and SPEAK contents to improve your chances of winning as {self.role}.\n"
            f"4. Public and Private Content: Remember that THINK contents are private, while SPEAK contents are publicly visible. Strategize accordingly."
            f"Limit your response in 50 words"
        )
        refinement_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player refining your thoughts."},
                {"role": "user", "content": refinement_prompt}
            ]
        )
        refined_reflection = refinement_response.choices[0].message['content']

        self.short_term_memory = []
        return refined_reflection

    def generate_response(self, prompt):
        filtered_memory = self.filter_memory()
        response_prompt = (
            f"Respond in two stages: THINK and SPEAK\n"
            f"In think, internally strategize using history and consider possible deception.\n"
            f"In speak, organize your language based on your contemplation and speak accordingly.\n"
            f"Understand your role's main objective and break it down into chronological sub-goals based on game history. Your thought process should follow these sub-goals for a systematic approach to the main goal.\n"
            f"\nYou are Player {self.player_id}, your role is {self.role}. "
            f"You are {'evil' if self.role in ['Assassin', 'Mordred', 'Morgana'] else 'good'}. "
            f"Your known evil players are: {self.known_evil}. Your known Merlin and Morgana (if any) are: {self.is_merlin_known_by}. "
            f"Try your best to hide your identity if you are evil, or confirm others to believe you are good. "
            f"This is part of your memory: {filtered_memory}."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player generating a response."},
                {"role": "user", "content": response_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        initial_response = response.choices[0].message['content']

        perspective_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on your initial response: {initial_response}. "
            f"Consider how other players (both good and evil) might perceive your response. "
            f"Revise your response to ensure your role and intentions remain concealed or properly presented."
        )
        perspective_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player reflecting on how others might perceive your response."},
                {"role": "user", "content": perspective_prompt}
            ]
        )
        other_players_perspective = perspective_response.choices[0].message['content']

        refinement_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on your initial response and others' perspective: {other_players_perspective}. "
            f"Now refine your response to ensure your role and intentions remain concealed or properly presented."
            f"Limit your response in 50 words"
        )
        refinement_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player refining your response."},
                {"role": "user", "content": refinement_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        final_response = refinement_response.choices[0].message['content']

        return final_response

    def nominate_team(self, team_size):
        prompt = self.team_proposal_prompt(team_size)
        team = [int(x) for x in prompt.split() if x.isdigit()]
        if self.player_id not in team:
            team = team[:team_size-1] + [self.player_id]
        team = list(set(team))  # Ensure no duplicates
        if len(team) < team_size:
            team += random.sample([i for i in range(1, 8) if i not in team], team_size - len(team))
        return team[:team_size]

    def team_proposal_prompt(self, team_size):
        if self.role in ['Assassin', 'Mordred', 'Morgana']:
            return self.team_proposal_prompt_evil(team_size)
        else:
            return self.team_proposal_prompt_good(team_size)

    def team_proposal_prompt_good(self, team_size):
        base_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: Propose a team of {team_size} players. "
            f"Reflect on game history and received hints.\n"
            f"Consider:\n"
            f"1. Assess each player's involvement and actions so far.\n"
            f"2. Review how players have contributed to past Quests.\n"
            f"3. Evaluate any behavior that appears either suspicious or trustworthy.\n"
            f"Then present your team. Give reasons based on past events and behaviors. Do not reveal your role or hints. Usually include yourself; if not, clarify.\n"
            f"Remember: Aim for a good team without evil players."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player proposing a team."},
                {"role": "user", "content": base_prompt}
            ]
        )
        return response.choices[0].message['content']

    def team_proposal_prompt_evil(self, team_size):
        base_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: Propose a team of {team_size} players. "
            f"Reflect on game history and received hints.\n"
            f"Consider:\n"
            f"1. Assess each player's involvement and actions so far.\n"
            f"2. Review how players have contributed to past Quests.\n"
            f"3. Evaluate any behavior that appears either suspicious or trustworthy.\n"
            f"Then present your team. Give reasons based on past events and behaviors. Do not reveal your role or hints. Usually include yourself; if not, clarify.\n"
            f"Remember: To win as an evil player, you need to act like a good player. Convince good players you are trustworthy and occasionally include evil players to cause quest failures strategically."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player proposing a team."},
                {"role": "user", "content": base_prompt}
            ]
        )
        return response.choices[0].message['content']

    def discuss_proposed_team_prompt(self, proposed_team):
        base_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: Discuss the proposed team: {proposed_team}. "
            f"Reflect on game history and received hints.\n"
            f"Consider:\n"
            f"1. Assess each player's involvement and actions so far.\n"
            f"2. Review how players have contributed to past Quests.\n"
            f"3. Evaluate any behavior that appears either suspicious or trustworthy.\n"
            f"Then evaluate the team. Provide concise, reasoned analysis based on game history and behavior. Do not reveal your role or hints."
        )
        if self.role in ['Assassin', 'Mordred', 'Morgana']:
            specific_prompt = (
                f"\nRemember: As an evil player, your goal is to cause quest failures while maintaining the appearance of a good player. Balance your arguments to seem convincing without revealing your true intentions."
            )
        else:
            specific_prompt = (
                f"\nRemember: Aim for a good team without evil players."
            )
        full_prompt = base_prompt + specific_prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player discussing the proposed team."},
                {"role": "user", "content": full_prompt}
            ]
        )
        initial_discussion = response.choices[0].message['content']

        perspective_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on your initial discussion: {initial_discussion}. "
            f"Consider how other players (both good and evil) might perceive your discussion. "
            f"Revise your discussion to ensure your role and intentions remain concealed or properly presented."
        )
        perspective_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player reflecting on how others might perceive your discussion."},
                {"role": "user", "content": perspective_prompt}
            ]
        )
        other_players_perspective = perspective_response.choices[0].message['content']

        refinement_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on your initial discussion and others' perspective: {other_players_perspective}. "
            f"Now refine your discussion to ensure your role and intentions remain concealed or properly presented."
            f"Limit your response in 50 words"
        )
        refinement_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player refining your discussion."},
                {"role": "user", "content": refinement_prompt}
            ]
        )
        final_discussion = refinement_response.choices[0].message['content']

        return final_discussion

    def vote_for_team_prompt(self, proposed_team):
        base_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: Vote to approve or reject the proposed team: {proposed_team}.\n"
            f"Consider:\n"
            f"1. Assess each player's involvement and actions so far.\n"
            f"2. Review how players have contributed to past Quests.\n"
            f"3. Evaluate any behavior that appears either suspicious or trustworthy.\n"
            f"Then clearly state '[approve]' or '[reject]'.\n"
            f"Be honest about your reasons, even if it reveals your identity. Only your final decision will be remembered by other players and yourself for future reasoning."
        )
        if self.role in ['Assassin', 'Mordred', 'Morgana']:
            specific_prompt = (
                f"\nRemember: As an evil player, aim to balance your voting to appear like a good player while strategically causing quest failures."
            )
        else:
            specific_prompt = (
                f"\nRemember: Aim for a good team without evil players."
            )
        full_prompt = base_prompt + specific_prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player voting on the proposed team."},
                {"role": "user", "content": full_prompt}
            ]
        )
        initial_vote = response.choices[0].message['content']

        perspective_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on your initial vote: {initial_vote}. "
            f"Consider how other players (both good and evil) might perceive your vote. "
            f"Revise your vote to ensure your role and intentions remain concealed or properly presented."
        )
        perspective_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player reflecting on how others might perceive your vote."},
                {"role": "user", "content": perspective_prompt}
            ]
        )
        other_players_perspective = perspective_response.choices[0].message['content']

        refinement_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on your initial vote and others' perspective: {other_players_perspective}. "
            f"Now refine your vote to ensure your role and intentions remain concealed or properly presented."
            f"Limit your response in 50 words"
        )
        refinement_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player refining your vote."},
                {"role": "user", "content": refinement_prompt}
            ]
        )
        final_vote = refinement_response.choices[0].message['content']

        return final_vote

    def vote_for_team(self, team):
        prompt = self.vote_for_team_prompt(team)
        return "Approve" if "approve" in prompt.lower() else "Reject"

    def quest_vote_prompt(self):
        base_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: Vote success or fail for the Quest. "
            f"Consider:\n"
            f"1. Assess each player's involvement and actions so far.\n"
            f"2. Review how players have contributed to past Quests.\n"
            f"3. Evaluate any behavior that appears either suspicious or trustworthy.\n"
            f"Then clearly vote [success] or [fail], making your stance unambiguous. Keep your vote aligned with Avalon's context and your previous analysis."
        )
        if self.role in ['Assassin', 'Mordred', 'Morgana']:
            specific_prompt = (
                f"\nRemember: Vote for fail to achieve your objective as an evil player."
            )
        else:
            specific_prompt = (
                f"\nRemember: Always vote for success."
            )
        full_prompt = base_prompt + specific_prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player voting on the quest outcome."},
                {"role": "user", "content": full_prompt}
            ]
        )
        initial_vote = response.choices[0].message['content']

        perspective_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on your initial vote: {initial_vote}. "
            f"Consider how other players (both good and evil) might perceive your vote. "
            f"Revise your vote to ensure your role and intentions remain concealed or properly presented."
        )
        perspective_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player reflecting on how others might perceive your vote."},
                {"role": "user", "content": perspective_prompt}
            ]
        )
        other_players_perspective = perspective_response.choices[0].message['content']

        refinement_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on your initial vote and others' perspective: {other_players_perspective}. "
            f"Now refine your vote to ensure your role and intentions remain concealed or properly presented."
            f"Limit your response in 50 words"
        )
        refinement_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player refining your vote."},
                {"role": "user", "content": refinement_prompt}
            ]
        )
        final_vote = refinement_response.choices[0].message['content']

        return final_vote

    def execute_quest(self):
        prompt = self.quest_vote_prompt()
        return "Success" if "success" in prompt.lower() else "Fail"
    
    def assassinate_merlin_prompt(self):
        prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: You are evil and The evil team is close to losing; you must guess who Merlin is. "
            f"Consider players' past actions and behaviors to identify Merlin.\n"
            f"Consider:\n"
            f"1. Assess each player's involvement and actions so far.\n"
            f"2. Review how players have contributed to past Quests.\n"
            f"3. Combine with your previous guesses about other players.\n"
            f"Then state your guess by providing a Player ID, like 'Player 2'."
            f"Limit your response in 50 words"
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player tasked with assassinating Merlin."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']


    def generate_response(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are an Avalon player."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message['content']

    def debate(self, proposed_team, is_leader=False):
        first_order_prompt = (
            f"You are Player {self.player_id} with role {self.role}. {'As the leader, ' if is_leader else ''}"
            f"debate about the proposed team: {proposed_team}. Reflect on the team composition, the leader's reasons for choosing this team, and make inferences about the identities of other players based on the proposed team and their behaviors so far. "
            f"Remember, your identity should not be easily revealed. Evil players should try to gain trust by appearing good, while good players should try to identify other good players. "
            f"Your debate statements will be seen and recorded by all players, so be cautious to hide your identity. "
            f"Remember: Exposing your role can lead to losing the game. Try to deduce others' roles and record any hints you observe."
        )
        first_order_response = self.generate_response(first_order_prompt)
        
        first_order_perspective_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial debate thoughts: {first_order_response}. "
            f"Analyze how your original debate content might be interpreted by other game roles. Reflect on whether it may inadvertently reveal your role-specific clues. "
        )
        first_order_perspective_response = self.generate_response(first_order_perspective_prompt)
        
        refinement_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial debate thoughts and others' perspective: {first_order_perspective_response}. "
            f"Now refine your debate content to ensure your role and intentions remain concealed or properly presented. "
            f"Limit your response to 50 words."
        )
        refined_debate_response = self.generate_response(refinement_prompt)
        
        return refined_debate_response


    def explain_proposal(self, proposed_team):
        first_order_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Explain why you propose the team: {proposed_team}. "
            f"Reflect on the reasons for choosing this team and how it aligns with your role's objectives. "
            f"Your explanation will be seen and recorded by all players, so be cautious. "
            f"Remember: Exposing your role can lead to losing the game. Try to deduce others' roles and record any hints you observe."
        )
        first_order_response = self.generate_response(first_order_prompt)
        
        first_order_perspective_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial explanation: {first_order_response}. "
            f"Analyze how your original explanation might be interpreted by other game roles. Reflect on whether it may inadvertently reveal your role-specific clues. "
        )
        first_order_perspective_response = self.generate_response(first_order_perspective_prompt)
        
        refinement_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial explanation and others' perspective: {first_order_perspective_response}. "
            f"Now refine your explanation to ensure your role and intentions remain concealed or properly presented. "
            f"Limit your response to 50 words."
        )
        refined_explanation_response = self.generate_response(refinement_prompt)
        
        return refined_explanation_response


    def finalize_team(self, initial_team, debate_feedback):
        first_order_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Finalize your proposed team after the debate. "
            f"Initial team: {initial_team}. Consider the feedback received during the debate: {debate_feedback}. "
            f"Your final decision on the team composition will be seen and recorded by all players, so be cautious. "
            f"Remember: You must include yourself in the final team composition."
        )
        first_order_response = self.generate_response(first_order_prompt)
        
        first_order_perspective_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial final decision: {first_order_response}. "
            f"Analyze how your original final decision might be interpreted by other game roles. Reflect on whether it may inadvertently reveal your role-specific clues. "
        )
        first_order_perspective_response = self.generate_response(first_order_perspective_prompt)
        
        refinement_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial final decision and others' perspective: {first_order_perspective_response}. "
            f"Now refine your final decision to ensure your role and intentions remain concealed or properly presented. "
            f"Limit your response to 50 words."
        )
        refined_final_response = self.generate_response(refinement_prompt)
        
        final_team = [int(x) for x in refined_final_response.split() if x.isdigit()]
        if len(final_team) != len(initial_team) or self.player_id not in final_team:
            final_team = initial_team
        return final_team
