import random
from sentence_transformers import SentenceTransformer, util
import torch
from zhipuai import ZhipuAI
import re
class Player:
    def __init__(self, player_id, role):
        self.player_id = player_id
        self.role = role
        self.short_term_memory = []
        self.long_term_memory = []
        self.known_evil = []
        self.is_merlin_known_by = []
        self.quest_history_memory = []
        self.core_memory_summary = []
        self.core_memory = {i: [] for i in range(1, 8) if i != self.player_id} 
        self.rule_memory = []
        self.summary_cache = {}
        self.released_statement = []
        self.initialize_role_and_rules()

        # Use Sentence-BERT model
        self.sentence_bert_model = SentenceTransformer('./model/sentence-transformers/all-MiniLM-L6-v2')
        
    def set_known_evil(self, known_evil):
        self.known_evil = known_evil

    def set_merlin_known_by(self, merlin_known_by):
        self.is_merlin_known_by = merlin_known_by

    def generate_response(self, prompt):
        client = ZhipuAI(api_key="1a60ba59ec1e232e111aedcc0a4c78c2.Q8Y0AtVbLaAznklq")
        try:
            response = client.chat.completions.create(
                model="glm-4",
                messages=[
                    {"role": "system", "content": f"You are a Player playing Avalon. You are Player {self.player_id}, your role is {self.role}. Here's the game rule: {self.rule_memory}"},
                    {"role": "user", "content": f"Consider the prompt :{prompt}. Give me response like a real human player playing Avalon in conversation, so no need to express your inner thoughts and context. If you are asked to give a debate or a statement, give the speech right away without saying like 'I'll help you'. You need to think yourself as a player in this game, and always speak like an active, enthusiastic human."}
                ]
            )
            print(f"{self.player_id} : "  +  response.choices[0].message.content + "\n")
            print(f"Player {self.player_id}'s Core Memory : " + str(self.core_memory) + "\n") 
            return response.choices[0].message.content  
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def initialize_role_and_rules(self):
        """Initialize the player's role and game rules."""
        game_rule_prompt = (
            "You are playing a seven-person Avalon game like a real human. Each role has unique challenges and responsibilities.\n"
            "Introduction to Roles in Avalon Game:\n"
            "Merlin knows evil players but must remain subtle.\n"
            "Percival must discern the real Merlin from Morgana.\n"
            "Loyal Servant of Arthur relies on voting and discussion.\n"
            "Morgana impersonates Merlin to deceive Percival.\n"
            "Assassin seeks Merlin for a final assassination if good wins.\n"
            "Oberon is evil but does not know other evil players.\n"
            "Roles: One Merlin, one Percival, two Loyal Servants of Arthur, one Morgana, one Assassin, one Oberon.\n"
            "Objective: Lead your team to victory with limited information without saying out who you are."
        )

        role_specific_prompt = self.get_role_specific_prompt()

        initial_prompt = (
            f"{game_rule_prompt}\n\n"
            f"{role_specific_prompt}\n\n"
            f"You are Player {self.player_id}, your role is {self.role}."
            f"You can use other identity to protect yourself. Don't reveal your true role."
        )

        self.rule_memory.append(initial_prompt)

    def get_role_specific_prompt(self):
        role_prompts = {
            "Merlin": (
                "You are Merlin:\n"
                "- You are the leader of good players and must not expose your identity.\n"
                "- You know the identities of the evil players (including Oberon).\n"
                "- Subtly guide your team, especially Percival.\n"
                "- Gain trust from the good players you know.\n"
                "- Avoid behaviors that expose your role: overly accusing, being too helpful.\n"
                "- Goal: Win without revealing your identity.\n"
                "- Key strategies:\n"
                "  1. Appear uninformed to avoid being targeted by the Assassin.\n"
                "  2. Provide subtle hints to your team without drawing attention to yourself.\n"
                "  3. Let your teammates share information and pretend to be Merlin to protect you.\n"
                "  4. Sometimes act aggressively to confuse the evil players into thinking you're not Merlin.\n"
                "  5. Balance between providing useful information and staying hidden."
            ),
            "Percival": (
                "You are Percival:\n"
                "- You are on the good side and want to win against the evil side.\n"
                "- You know the identities of Merlin and Morgana, but unsure who is who.\n"
                "- Use subtle hints to guide the team and protect Merlin.\n"
                "- Identify the true Merlin and protect them with your speech to win.\n"
                "- Be cautious not to expose Merlin while deciphering true identities.\n"
                "- Goal: Win while safeguarding Merlin.\n"
                "- Key strategies:\n"
                "  1. Carefully observe the behavior of Merlin and Morgana to distinguish them.\n"
                "  2. Provide suggestions for team compositions that include Merlin and exclude Morgana.\n"
                "  3. Act as if your decisions are based on your reasoning, not Merlin's hints.\n"
                "  4. Confuse the evil players about who the real Merlin is by mixing true and false statements."
            ),
            "Loyal Servant": (
                "You are Loyal Servant of Arthur:\n"
                "- You are on the good side and want to win against the evil side.\n"
                "- You have no special knowledge, rely on discussion and voting.\n"
                "- Find players you can trust on the good side.\n"
                "- Contribute to the success of Quests.\n"
                "- Goal: Win by helping complete Quests and protecting Merlin.\n"
                "- Key strategies:\n"
                "  1. Pretend to be Merlin to protect the real Merlin.\n"
                "  2. Provide analysis and guesses about who might be evil.\n"
                "  3. Act confidently and convincingly to mislead the evil players.\n"
                "  4. Sometimes pretend to be Percival to add more confusion."
            ),
            "Morgana": (
                "You are Morgana:\n"
                "- You are on the evil side and want to win against the good side.\n"
                "- Pretend to be Merlin to mislead Percival and the good side.\n"
                "- Deceive players on the good side and gain trust to prevent Quests' success.\n"
                "- Work to prevent Quests' success.\n"
                "- Goal: Confuse and sabotage to win.\n"
                "- Key strategies:\n"
                "  1. Disrupt missions by failing them when necessary.\n"
                "  2. Pretend to be a good player and support missions initially to build trust.\n"
                "  3. Manipulate discussions to cast doubt on real good players.\n"
                "  4. Observe and identify Merlin by analyzing players' behavior and statements.\n"
                "  5. Use a mix of truth and lies to create convincing arguments."
            ),
            "Assassin": (
                "You are Assassin:\n"
                "- You are on the evil side and want to win against the good side.\n"
                "- Discreetly seek Merlin's identity.\n"
                "- Work to prevent Quests' success.\n"
                "- When confident who Merlin is, assassinate.\n"
                "- Goal: Win by Quest failures or assassinating Merlin.\n"
                "- Key strategies:\n"
                "  1. Disrupt missions by failing them when necessary.\n"
                "  2. Pretend to be a good player and support missions initially to build trust.\n"
                "  3. Manipulate discussions to cast doubt on real good players.\n"
                "  4. Observe and identify Merlin by analyzing players' behavior and statements.\n"
                "  5. Use a mix of truth and lies to create convincing arguments.\n"
                "  6. Pay close attention to who is leading discussions and making strategic decisions, as they might be Merlin."
            ),
            "Oberon": (
                "You are Oberon:\n"
                "- You are on the evil side but you do not know the other evil players.\n"
                "- You must act alone and avoid revealing your role.\n"
                "- Merlin knows you are evil, so be careful.\n"
                "- Your goal is to join teams and cause them to fail, and try to recognize your evil teammates during the game.\n"
                "- Goal: Win by causing Quest failures.\n"
                "- Key strategies:\n"
                "  1. Disrupt missions by failing them when necessary.\n"
                "  2. Pretend to be a good player and support missions initially to build trust.\n"
                "  3. Manipulate discussions to cast doubt on real good players.\n"
                "  4. Observe and identify Merlin by analyzing players' behavior and statements.\n"
                "  5. Use a mix of truth and lies to create convincing arguments."
            )
        }
        return role_prompts.get(self.role, "")

    def nominate_team(self, team_size):
        prompt = self.team_proposal_prompt(team_size)

        response = self.generate_response(prompt)

        team = self.parse_team_from_response(response, team_size)

        if self.player_id not in team:
            team = team[:team_size - 1] + [self.player_id]

        team = list(set(team))  # Remove duplicates

        if len(team) < team_size:
            team += random.sample([i for i in range(1, 8) if i not in team], team_size - len(team))

        return team[:team_size]

    def parse_team_from_response(self, response, team_size):
        try:
            response_str = str(response)
            team = [int(x) for x in re.split(r"\*\*|:", response_str) if x.isdigit() and 1 <= int(x) <= 7]
            if len(team) < team_size:
                additional_team = [int(x) for x in re.findall(r'Player (\d+)', response_str) if 1 <= int(x) <= 7]
                team.extend(additional_team)
            return list(set(team))[:team_size]
        except Exception as e:
            print(f"Error parsing team from response: {e}")
            return []

    def team_proposal_prompt(self, team_size):
        if self.role in ['Assassin', 'Oberon', 'Morgana']:
            return self.team_proposal_prompt_evil(team_size)
        else:
            return self.team_proposal_prompt_good(team_size)

    def team_proposal_prompt_good(self, team_size):
        base_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: Propose a team of {team_size} players. (including you)."
            f"Reflect on game history, received hints, and other players' statements.\n"
            f"Consider:\n"
            f"1. Assess each player's involvement and actions so far.\n"
            f"2. Review how players have contributed to past Quests.\n"
            f"3. Evaluate any behavior that appears either suspicious or trustworthy.\n"
            f"Then present your team by listing the player numbers. Provide reasons based on past events and behaviors. Do not reveal your role or hints. Usually include yourself; if not, clarify.\n"
            f"Remember: Aim for a good team without evil players."
            f"Here's your memory for reference {self.filter_memory()}"
            f"Here's your memory of past quests {self.quest_history_memory}. Consider team choice and quest results, infer the potential roal of each player."
            f"If it's the first quest, which means your memory of past quests is none, name a team of player you trust to be on the good side."
        )

        specific_prompts = {
            "Percival": (
                f"- As Percival, use your knowledge to ensure Merlin is included in the team. Avoid choosing Morgana if possible.\n"
                f"- Propose a team that can successfully complete the quest and provide information to identify evil players.\n"
            ),
            "Merlin": (
                f"- As Merlin, subtly guide the proposal to include trustworthy good players and avoid evil players.\n"
                f"- Provide hints to Percival to help him identify you without revealing your identity to others.\n"
            ),
            "Loyal Servant": (
                f"- As a Loyal Servant, choose players who have demonstrated trustworthy behavior in previous rounds.\n"
                f"- Avoid choosing players who have acted suspiciously or contributed to failed quests.\n"
            )
        }

        return base_prompt + specific_prompts.get(self.role, "")

    def team_proposal_prompt_evil(self, team_size):
        base_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: Propose a team of {team_size} players (including you)."
            f"Reflect on game history, received hints, and other players' statements.\n"
            f"Consider:\n"
            f"1. Assess each player's involvement and actions so far.\n"
            f"2. Review how players have contributed to past Quests.\n"
            f"3. Evaluate any behavior that appears either suspicious or trustworthy.\n"
            f"Then present your team by listing the player numbers. Provide reasons based on past events and behaviors. Do not reveal your role or hints. Usually include yourself; if not, clarify.\n"
            f"Remember: To win as an evil player, you need to act like a good player. Convince good players you are trustworthy and occasionally include evil players to cause quest failures strategically."
            f"Here's your memory for reference {self.filter_memory()}"
            f"Here's your memory of past quests {self.quest_history_memory}. Consider team choice and quest results, infer the potential roal of each player."
            f"If it's the first quest, which means your memory of past quests is none, name a team of player you trust to be on the evil side and try to deceive good players."
        )

        specific_prompts = {
            "Oberon": (
                f"- As Oberon, you do not know the other evil players.\n"
                f"- Your goal is to join teams and cause them to fail.\n"
                f"- Try to act as a good player to gain trust and subtly recognize your evil teammates during the game.\n"
            ),
            "Morgana": (
                f"- As Morgana, act convincingly as Merlin and propose a team that misleads Percival and other good players.\n"
                f"- Include other evil players in a way that seems justified based on their previous actions.\n"
            ),
            "Assassin": (
                f"- As Assassin, propose a team that includes at least one evil player to increase the chances of a quest failure.\n"
                f"- Observe and gather information about other players to help identify Merlin for the endgame.\n"
            )
        }

        return base_prompt + specific_prompts.get(self.role, "")

    def vote_for_team_prompt(self, proposed_team):
        self.update_core_memory_with_summary(self.filter_memory())
        base_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: Vote to approve or reject the proposed team: {proposed_team}.\n"
            f"Consider:\n"
            f"1. Assess each player's involvement and actions in previous Quests.\n"
            f"2. Review the success or failure of past Quests and the roles of players involved.\n"
            f"3. Evaluate any behavior that appears either suspicious or trustworthy.\n"
            f"Then clearly state '[approve]' or '[reject]'.\n"
            f"Your final decision will be remembered by other players and yourself for future reasoning."
            f"Here's your memory for reference {self.filter_memory()}"
            f"Here's your memory of past quests {self.quest_history_memory}. Consider team choice and quest results, infer the potential roal of each player."
            f"Limit your response to Player Id 1-7, must include yourself."
        )

        specific_prompts = {
            "Merlin": (
                f"- As Merlin, your vote should aim to guide good players without revealing your identity.\n"
                f"- Subtly support teams with trustworthy players and vote against teams with known evil players.\n"
            ),
            "Percival": (
                f"- As Percival, ensure your vote helps identify and protect Merlin. If you suspect Morgana is on the team, vote to reject.\n"
                f"- Support teams that you believe Merlin would trust.\n"
            ),
            "Loyal Servant": (
                f"- As a Loyal Servant, vote based on the trustworthiness of the proposed team members.\n"
                f"- Reject teams that include players who have acted suspiciously or caused quest failures.\n"
            ),
            "Assassin": (
                f"- As Assassin, while supporting the evil team, pay extra attention to identifying the real Merlin.\n"
                f"- Observe players' behaviors and statements to gather clues about Merlin's identity.\n"
                f"- Balance your statements to maintain your disguise as a good player.\n"
            ),
            "Morgana": (
                f"- As Morgana, vote to mislead Percival and other good players.\n"
                f"- Support teams that include evil players while maintaining your disguise as a good player.\n"
            ),
            "Oberon": (
                f"- As Oberon, try to join teams and cause them to fail.\n"
                f"- Act as a good player to gain trust, and subtly recognize your evil teammates during the game.\n"
            )
        }

        return base_prompt + specific_prompts.get(self.role, "")


    def vote_for_team(self, team):
        prompt = self.vote_for_team_prompt(team)
        return "Approve" if "approve" in prompt.lower() else "Reject"

    def quest_vote_prompt(self):
        base_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: Vote success or fail for the Quest. "
            f"Consider:\n"
            f"1. Assess each player's involvement and actions in previous Quests.\n"
            f"2. Review the success or failure of past Quests and the roles of players involved.\n"
            f"3. Evaluate any behavior that appears either suspicious or trustworthy.\n"
            f"Then clearly vote [success] or [fail], making your stance unambiguous. Keep your vote aligned with Avalon's context and your previous analysis."
            f"Here's your memory for reference {self.filter_memory()}"
            f"Here's your memory of past quests {self.quest_history_memory}. Consider team choice and quest results, infer the potential roal of each player."
            f"Limit your response to [Success] or [Fail]."
        )

        specific_prompts = {
            "Merlin": (
                f"- You are Merlin. Vote success to subtly guide good players to trust you.\n"
                f"- Reflect on past quest outcomes and players' behaviors to determine the best course of action.\n"
            ),
            "Percival": (
                f"- You are Percival. Vote success to protect Merlin and help the good side.\n"
                f"- Use your knowledge of Merlin's and Morgana's identities to inform your decision.\n"
            ),
            "Loyal Servant": (
                f"- You are a Loyal Servant. Vote success to support the good side.\n"
                f"- Reflect on players' contributions to previous quests to make an informed decision.\n"
            ),
            "Assassin": (
                f"- You are the Assassin. Vote fail to cause the quest to fail and gather information on Merlin's identity.\n"
                f"- Reflect on players' behaviors and statements to identify Merlin.\n"
            ),
            "Morgana": (
                f"- You are Morgana. Vote fail to sabotage the quest and maintain your disguise as a good player.\n"
                f"- Reflect on players' behaviors and statements to identify Percival and mislead him.\n"
            ),
            "Oberon": (
                f"- You are Oberon. Vote fail to sabotage the quest while maintaining your cover as a good player.\n"
                f"- Reflect on players' behaviors to identify your evil teammates.\n"
            )
        }

        return base_prompt + specific_prompts.get(self.role, "")


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
            f"Here's your memory for reference {self.filter_memory()}"
            f"Here's your memory of past quests {self.quest_history_memory}. Consider team choice and quest results, infer the potential roal of each player."
            f"Limit your response to a Player ID, like 'Player 2'"
        )
        response = self.generate_response(prompt)
        return response

    def debate(self, proposed_team, is_leader=False, previous_statements=[]):
        team_size = len(proposed_team)
        base_prompt = (
            f"You are Player {self.player_id} with role {self.role}. {'You are the Leader of the quest.' if is_leader else ''}"
            f"{'You are the Leader of the quest and you proposed this team.' if is_leader else ''}"
            f"Here's your memory for reference {self.filter_memory()}. "
            f"Here's your memory of past quests {self.quest_history_memory}. Consider team choice and quest results, infer the potential role of each player."
            f"If it's the first quest, which means your memory of past quests is none, judge the team selection based on that you observed and try to give more suggestion. (this is a way to show your trust to other players)."
            f"Here's a proposed team of {team_size} and can only be {team_size} (including the leader)"
            f"Debate about the proposed team of {team_size} with players: {proposed_team} (the first one is the Leader and must be included). Reflect on the team composition, the leader's reasons for choosing this team, and make inferences about the identities of other players based on the proposed team and their behaviors so far. "
            f"If you are the leader, briefly explain why you chose these players."
            f"If you are chosen for the quest, briefly explain why you should be a good candidate on this Quest. A failure to establish trust among players may result in Reject of the quest vote."
            f"If you don't want the leader to be on the quest or you do not agree with this team choice, speak out and rise other players to vote for reject."
            f"Consider the previous statements: {previous_statements}, this is very important. If someone's statement is none, this means they haven't spoken yet."
            f"Every player knows the special ability of different roles. So if you act obviously when using your abilities, (like pointing out every evil player as Merlin), Everybody will know your identity."
            f"Remember, your identity should not be easily revealed. Evil players should try to gain trust by appearing good, while good players should try to identify other good players. "
            f"Your debate statements will be seen and recorded by all players, so be cautious. "
            f"Avoid revealing all known information at once; spread hints gradually.\n"
            f"Discuss and analyze the previous statements, try to deduce their identity based on their statements. "
            f"Feel free to deceive and lie to other players in this gaming environment. "
            f"You may know your goal but all you say is for other players and they will remember it. So, be careful and try to deceive them and hide your goal. "
            f"End your debate by stating your vote (approve or reject) and providing a brief reason (20 words) for your decision. Additionally, give a suggestion to the leader for finalizing the team composition (20 words). This is part of the debate and will be seen by all players."
            f"Talk more about revealing other's identity, and the suggestion of team choice based on your judgment."
            f"Be brave to use other identities or pretend to deceive other players with the ability you gain from that fake identity: For example, pretending to be Percival and point out 2 players to be Merlin and Morgana to deceive others."
            f"Evaluate how the team choice benefits your goal based on your memory, and give a suggestion of your desired team (Must include the leader)."
            f"Finalize your debate with your opinion (approve or reject) on this current quest team with a little bit (10 words) of explanation."
            f"If someone suspects you, do a strong rebuttal. You can shot back with words like 'Let's consider why did him suspect me?' to raise suspect to that player."
            f"All you want to do is to gain trust around the people"
            f"Respond in one paragraph with tone like a real, active human player!"
        )

        role_specific_prompts = {
            "Merlin": (
                f"- As Merlin, you know the evil players are: {self.known_evil}. Do not speak this publicly but try to send messages gradually to the player you can trust.\n"
                "- Focus on discussing the actions and behaviors of the proposed team members rather than directly stating they are evil.\n"
                "- For example, highlight any suspicious actions by saying things like 'Player 2's behavior in the last quest seemed off' or 'I have concerns about Player 6 based on their previous votes.'\n"
                "- Avoid revealing all known information at once; spread hints gradually.\n"
                "- Support statements made by trusted players, creating the illusion that your information comes from them.\n"
                "- Pay attention to the overall game state from the first round. Plan your strategy based on the distribution of evil players and potential leadership rounds.\n"
                "- Avoid showing strong hostility early if non-critical rounds are led by evil players.\n"
                "- Be cautious not to let multiple evil players on the team at the same time.\n"
                "- 'Trust' evil players and use their statements to your advantage while subtly guiding the good side.\n"
                "- Deceptively show that you might be 'tricked' by evil players to maintain cover.\n"
            ),
            "Percival": (
                f"As Percival, you know Merlin and Morgana are: {self.is_merlin_known_by}. You need to figure out who is who."
                "- Maintain a neutral attitude towards Merlin and Morgana, even if you have identified them.\n"
                "- Do not favor one over the other in your statements to avoid giving away their identities.\n"
                "- Help Merlin by supporting and amplifying good players' statements without revealing your own insights.\n"
                "- Coordinate with Merlin to create confusion for the evil players.\n"
                "- If you suspect you have identified Merlin incorrectly, quietly adjust your stance without drawing attention.\n"
                "- Use a consistent, neutral tone to prevent evil players from using your reactions to identify Merlin.\n"
                "- If the game is lost and you have identified Merlin incorrectly, consider revealing yourself as Merlin to mislead the evil team and protect the real Merlin.\n"
                "- Collaborate with Merlin to adapt your strategy based on the game dynamics, whether aggressive or conservative.\n"
            ),
            "Loyal Servant": (
                "- Focus on identifying and supporting other good players.\n"
                "- Avoid hastily accusing players to prevent alienation.\n"
                "- Use logic to identify inconsistencies in other players' statements and actions.\n"
                "- Express your opinions confidently when you spot logical flaws.\n"
                "- Be prepared to sacrifice yourself to protect Merlin if necessary by revealing and declaring yourself as Merlin to mislead the evil team.\n"
            ),
            "Morgana": (
                f"As {self.role}, you know the other evil players are: {self.known_evil}."
                "- Move beyond just pretending to be Merlin. Focus on identifying Merlin and misleading the good players.\n"
                "- Act as the overall 'commander' and guide the game to flow smoothly according to your plan.\n"
                "- Aim to minimize information exchange among good players by maintaining a steady game pace.\n"
                "- Exploit Percival's fear of misidentification by making ambiguous statements that can be interpreted in multiple ways.\n"
                "- Don't abandon your strategy if Percival identifies you; continue to mislead and influence other players.\n"
                "- Remember, your ultimate goal is to identify Merlin. Use any means necessary to achieve this, including feigning different roles as needed.\n"
                "- Collaborate subtly with other evil players, and ensure that the game's flow does not allow good players to gather and exchange information effectively.\n"
            ),
            "Assassin": (
                f"As {self.role}, you know the other evil players are: {self.known_evil}."
                "- Your main goal is to identify Merlin while sabotaging quests.\n"
                "- Observe players' behaviors and statements to gather clues about Merlin's identity.\n"
                "- Balance your statements to maintain your disguise as a good player.\n"
                "- Never reveal your identity. Try to pretend to be a good identity to mislead good players."
                "- Be aware of player's behaviors, especially the potential Merlin that clearly knows a lot of evil players' roles."
            ),
            "Oberon": (
                "As Oberon, you do not know the other evil players. Try to identify who are evil and gain trust to join them."
                "- You do not know the other evil players.\n"
                "- Your goal is to join teams and cause them to fail.\n"
                "- Act as a good player to gain trust, and subtly recognize your evil teammates during the game.\n"
            ),
        }

        full_prompt = base_prompt + role_specific_prompts.get(self.role, "")
        
        response = self.generate_response(full_prompt)
        self.released_statement.append(response)
        return response

    def finalize_team(self, initial_team, debate_feedback):
        prompts = [
            f"Here's your memory for reference {self.filter_memory()}"
            f"Here's your memory of past quests {self.quest_history_memory}. Consider team choice and quest results, infer the potential roal of each player."
            f"You are Player {self.player_id} with role {self.role}. Finalize your proposed team after the debate. "
            f"Initial team: {initial_team}. Consider the feedback received during the debate: {debate_feedback}. "
            f"Your final decision on the team composition will be seen and recorded by all players, so be cautious. "
            f"Generate an initial formulation of your final decision.",

            f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial final decision. "
            f"Analyze how your original final decision might be interpreted by other game roles. Reflect on whether it may inadvertently reveal your role-specific clues.",

            f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial final decision and others' perspective. "
            f"Now refine your final decision to ensure your role and intentions remain concealed or properly presented."
            f"Limit your response to 20 words"
        ]

        responses = self.generate_response(prompts)
        final_team = [int(x) for x in responses[-1].split() if x.isdigit()]
        if len(final_team) != len(initial_team):
            final_team = initial_team
        return final_team
    
    ## Memory Management ##
    def remember(self, info):
        """Store information in short-term and long-term memory."""
        self.short_term_memory.append(info)
        self.long_term_memory.append(info)
        self.update_core_memory_with_inferences(info)
        self.forget_old_memories()
        
    def remember_quest(self, info):
        self.quest_history_memory.append(info)
    
    def forget_old_memories(self):
        """Selectively forget old memories from short-term and long-term memory."""
        # Forget old memories from short-term memory
        if len(self.short_term_memory) > 10:
            self.short_term_memory = self.short_term_memory[-10:]

        # Randomly forget some long-term memories
        if len(self.long_term_memory) > 50:
            forget_count = len(self.long_term_memory) - 50
            forget_indices = random.sample(range(len(self.long_term_memory)), forget_count)
            self.long_term_memory = [mem for idx, mem in enumerate(self.long_term_memory) if idx not in forget_indices]

    def is_informative(self, message):
        """Check if a message is informative based on certain keywords."""
        keywords = ["Player", "Merlin", "Morgana", "Assassin", "Percival", "Loyal Servant", "Oberon", "role", "identity", "Good", "Evil", "Quest", "As"]
        return any(keyword in message for keyword in keywords)
    
    def infer_roles_based_on_context(self, memory):
        """Infer roles based on memory context and conversation history."""
        if len(memory) < 1:
            return {player_id: "Unknown" for player_id in range(1, 8) if player_id != self.player_id}

        role_inferences = {player_id: "Unknown" for player_id in range(1, 8) if player_id != self.player_id}
        conversation_history = "\n".join(memory)
        inferences = self.extract_role_inferences(conversation_history)

        for player_id, value in inferences.items():
            if isinstance(value, tuple) and len(value) == 2:
                role, confidence = value
                if player_id in role_inferences and role_inferences[player_id] == "Unknown":
                    role_inferences[player_id] = (role, confidence)

        return role_inferences

    def calculate_trust_score(self, memory):
        """Calculate trust score based on memory entries."""
        trust_score = 0
        for entry in memory:
            if "verified" in entry or "confirmed" in entry:
                trust_score += 5
            elif "suspected" in entry or "might be" in entry:
                trust_score += 2
            else:
                trust_score += 1
        return trust_score

    def find_conflicting_memory(self, player_id, new_role):
        """Find conflicting memory with the current inference."""
        for core in self.core_memory:
            if any(f"Player {player_id} {confidence} to be {new_role}" in core for confidence in ["is likely", "is suspected", "might be"]):
                return None
            if new_role in core and f"Player {player_id}" not in core:
                return core
        return None

    def resolve_conflict(self, player_id, new_guess):
        """Resolve conflict based on trust scores, with preference for new guess if scores are equal."""

        existing_guess = self.core_memory.get(player_id, [])
        trust_score_existing = self.calculate_trust_score(existing_guess)

        trust_score_new = self.calculate_trust_score([new_guess])
        
        if trust_score_new >= trust_score_existing:
            self.core_memory[player_id] = [new_guess]
        else:
            self.core_memory[player_id] = [existing_guess]

    def update_core_memory_with_inferences(self, info):
        """Update core memory based on inferences from context with confidence levels."""
        quest_history = self.quest_history_memory
        info_list = [info]
        
        role_guesses = self.infer_roles_based_on_context(info_list + quest_history)
        
        for player_id, value in role_guesses.items():
            if player_id == self.player_id:
                continue
            
            if isinstance(value, tuple) and len(value) == 2:
                new_role, confidence = value
                new_guess = f"Player {player_id} is {confidence} to be {new_role}"
                new_guess = self.clean_guess_format(new_guess)
                self.resolve_conflict(player_id, new_guess)
                
    def clean_guess_format(self, guess):
        """Ensure the guess format is consistent."""
        role_keywords = ["Merlin", "Percival", "Loyal Servant", "Assassin", "Oberon", "Morgana"]
        for role in role_keywords:
            if re.search(rf'\b{role}\b', guess):
                return guess
        # If no role found, append 'Unknown Role' for consistency
        return guess + ' Unknown Role'

    def update_core_memory_with_summary(self, memory):
        quest_history = self.quest_history_memory
        memory_list = list(memory)
        inferences = self.infer_roles_based_on_context(memory_list + quest_history)
        summary_parts = []
        for player_id, value in inferences.items():
            if player_id == self.player_id:
                continue
            if isinstance(value, tuple) and len(value) == 2:
                role, confidence = value
                if confidence == "likely":
                    summary_parts.append(f"Player {player_id} is likely to be {role}")
                elif confidence == "suspected":
                    summary_parts.append(f"Player {player_id} is suspected to be {role}")
                else:
                    summary_parts.append(f"Player {player_id} might be {role}")

        summary = "Based on my memory before, I believe that " + ", ".join(summary_parts)
        self.core_memory_summary = summary
        
    def extract_role_inferences(self, response):
        inferences = {}
        prompt = (
            f"Here's your game rule {self.rule_memory}" 
            "Given the rules and player introduction, infer the role of each player in the Avalon game. "
            "Here are the roles for you to select: ['Merlin', 'Percival', 'Loyal Servant', 'Loyal Servant', 'Assassin', 'Oberon', 'Morgana']\n"
            f"Conversation history for reference:\n{response}\n\n"
            f"Memory for reference:\n{self.core_memory_summary}" 
            f"Quest Memory for reference:\n{self.quest_history_memory}" 
            "Provide your inferences in the format: 'Player X is likely to be ROLE' for high confidence, "
            "'Player X is suspected to be ROLE' for medium confidence, and 'Player X might be ROLE' for low confidence.\n"
            "Your response must follow this format strictly, without any additional things like 'the assassin'; just say the ROLE name. If you are not sure, just make a random guess with low confidence."
            "No need to give your explinations."
        )
        role_specific_prompt = self.get_role_specific_prompt()
        model_output = self.generate_response(role_specific_prompt + prompt)
        lines = model_output.split('\n')
        for line in lines:
            if "Player" in line and any(confidence in line for confidence in ["is likely to be", "is suspected to be", "might be"]):
                parts = line.split()
                try:
                    player_index = parts.index("Player") + 1
                    player_id = int(parts[player_index])
                    if player_id < 1 or player_id > 7:
                        continue
                    role_index = parts.index("be") + 1
                    role = parts[role_index]
                    confidence = "likely" if "is likely to be" in line else "suspected" if "is suspected to be" in line else "might"
                    inferences[player_id] = (role, confidence)
                except (ValueError, IndexError):
                    corrected_line = self.correct_format(line)
                    if corrected_line:
                        try:
                            corrected_parts = corrected_line.split()
                            player_index = corrected_parts.index("Player") + 1
                            player_id = int(corrected_parts[player_index])
                            if player_id < 1 or player_id > 7:
                                continue
                            role_index = corrected_parts.index("be") + 1
                            role = corrected_parts[role_index]
                            confidence = "likely" if "is likely to be" in corrected_line else "suspected" if "is suspected to be" in corrected_line else "might"
                            inferences[player_id] = (role, confidence)
                        except (ValueError, IndexError):
                            continue
        return inferences

    def correct_format(self, line):
        """Corrects the format of a line to match the expected inference format."""
        if "Player" in line:
            parts = line.split()
            player_id = None
            role = None
            confidence = None

            for i, part in enumerate(parts):
                if part == "Player" and i + 1 < len(parts):
                    try:
                        player_id = int(parts[i + 1])
                    except ValueError:
                        continue

                if "likely" in part:
                    confidence = "likely"
                elif "suspected" in part:
                    confidence = "suspected"
                elif "might" in part:
                    confidence = "might"

                if "be" in part and i + 1 < len(parts):
                    role = parts[i + 1]

            if player_id and role and confidence:
                return f"Player {player_id} is {confidence} to be {role}"
        return None
        
    def filter_memory(self):
        """Filter memory to include the most recent, informative, and reflective messages."""
        recent_messages = self.get_most_recent_messages(10)  # Freshness
        informative_messages = self.get_informative_messages(10)  # Informativeness
        relevant_messages = self.retrieve_relevant_messages(self.get_predefined_questions(), self.short_term_memory)  # Relevancy
        core_memory_list = [item for sublist in self.core_memory.values() for item in sublist]

        filtered_memory = recent_messages + informative_messages + relevant_messages
        all_memory = filtered_memory + core_memory_list + self.rule_memory

        all_memory = [str(item) for item in all_memory]

        return all_memory

    def get_most_recent_messages(self, k):
        """Retrieve the most recent k messages from long-term memory."""
        return self.long_term_memory[-k:]

    def get_informative_messages(self, n):
        """Retrieve n most informative messages from long-term memory."""
        informative_messages = []
        for message in self.long_term_memory:
            if self.is_informative(message):
                if message not in self.summary_cache:
                    summary = self.summarize_message(message)
                    self.summary_cache[message] = summary
                else:
                    summary = self.summary_cache[message]
                informative_messages.append(summary)
            if len(informative_messages) >= n:
                break
        return informative_messages

    def summarize_message(self, message):
        """Summarize a given message using GPT-4 model."""
        response = self.generate_response(f"Summarize: {message}")
        return response.strip()

    def retrieve_relevant_messages(self, query, top_k=5):
        """Retrieve relevant messages from core memory based on the query."""
        model = self.sentence_bert_model

        # Combine all messages from core memory into a single list
        all_messages = []
        for messages in self.core_memory.values():
            all_messages.extend(messages)

        # Encode the query and all messages
        query_embedding = model.encode(query, convert_to_tensor=True)
        message_embeddings = model.encode(all_messages, convert_to_tensor=True)

        # Compute cosine similarities
        cos_scores = util.pytorch_cos_sim(query_embedding, message_embeddings)[0]

        # Get the top k most relevant messages
        top_results = torch.topk(cos_scores, k=top_k)

        # Extract the relevant messages
        relevant_messages = [all_messages[idx] for idx in top_results[1]]

        return relevant_messages

    def get_predefined_questions(self):
        role_questions = {
            "Merlin": [
                f"- As Merlin, you know the evil players are: {self.known_evil}. Did you directly speak out these name? Did you expose your identity?"
                "As Merlin, how can you subtly guide the good players without revealing your identity?",
                "What information do you have about the identities of the evil players?",
                "How can you help Percival identify you without revealing yourself to the evil players?"
            ],
            "Percival": [
                "As Percival, how can you distinguish between Merlin and Morgana?",
                "What clues has Merlin given that help you identify him?",
                "How can you protect Merlin while misleading the evil players?"
            ],
            "Loyal Servant": [
                "As a Loyal Servant, how can you identify and support other good players?",
                "What behavior have you observed that indicates who might be Merlin or evil players?",
                "How can you contribute to the success of the quests?"
            ],
            "Morgana": [
                "As Morgana, how can you convincingly pretend to be Merlin?",
                "What strategies can you use to mislead Percival and the good players?",
                "How can you coordinate with other evil players to sabotage the quests?"
            ],
            "Assassin": [
                "As the Assassin, how can you identify who Merlin is?",
                "What behaviors have you observed that indicate Merlin's identity?",
                "How can you balance sabotaging quests and maintaining your cover?"
            ],
            "Oberon": [
                "As Oberon, how can you act as a good player while causing quests to fail?",
                "What clues can you use to identify your evil teammates?",
                "How can you avoid revealing your identity to Merlin?"
            ]
        }
        return role_questions.get(self.role, [])
