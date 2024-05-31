import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer, util

class Player:
    def __init__(self, player_id, role, experience_pool):
        self.player_id = player_id
        self.role = role
        self.short_term_memory = []
        self.long_term_memory = []
        self.known_evil = []
        self.is_merlin_known_by = []
        self.core_memory = []
        self.rule_memory = []
        self.summary_cache = {}
        self.experience_pool = experience_pool
        self.initialize_role_and_rules()

        self.local_model_path = "/Users/davidwang/Documents/GitHub/LLM_GAME/model/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, torch_dtype=torch.float32)
        self.sentence_bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def set_known_evil(self, known_evil):
        self.known_evil = known_evil

    def set_merlin_known_by(self, merlin_known_by):
        self.is_merlin_known_by = merlin_known_by

    def remember(self, info):
        """Store information in short-term and long-term memory."""
        self.short_term_memory.append(info)
        self.long_term_memory.append(info)
        self.update_core_memory_with_inferences(info)
        self.forget_old_memories()  # Invoke forgetting mechanism

    def is_informative(self, message):
        """Check if a message is informative based on certain keywords."""
        keywords = ["Player", "Merlin", "Morgana", "Assassin", "Percival", "Loyal Servant", "Oberon", "role", "identity", "Good", "Evil", "Quest", "As"]
        return any(keyword in message for keyword in keywords)

    def infer_roles_based_on_context(self, memory):
        """Infer roles based on memory context and conversation history."""
        role_inferences = {player_id: "Unknown" for player_id in range(1, 8) if player_id != self.player_id}
        conversation_history = "\n".join(memory)
        inferences = self.extract_role_inferences(conversation_history)

        for player_id, (role, confidence) in inferences.items():
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

    def resolve_conflict(self, conflicting_memory, new_guess):
        """Resolve conflict based on trust scores, with preference for new guess if scores are equal."""
        trust_score_conflicting = self.calculate_trust_score([conflicting_memory])
        trust_score_new = self.calculate_trust_score([new_guess])
        if trust_score_new >= trust_score_conflicting:
            self.core_memory.remove(conflicting_memory)
            self.core_memory.append(new_guess)
        else:
            self.core_memory.append(f"Conflicting guess: {new_guess}")

    def update_core_memory_with_inferences(self, info):
        """Update core memory based on inferences from context with confidence levels."""
        role_guesses = self.infer_roles_based_on_context([info])
        for player_id, (new_role, confidence) in role_guesses.items():
            if confidence == "likely":
                new_guess = f"Player {player_id} is likely to be {new_role}"
            elif confidence == "suspected":
                new_guess = f"Player {player_id} is suspected to be {new_role}"
            else:
                new_guess = f"Player {player_id} might be {new_role}"

            conflicting_memory = self.find_conflicting_memory(player_id, new_role)
            if conflicting_memory:
                self.resolve_conflict(conflicting_memory, new_guess)
            else:
                self.core_memory.append(new_guess)

    def extract_role_inferences(self, response):
        """Extract role inferences from model response with confidence levels."""
        inferences = {}
        prompt = (
            "Given the following conversation history, infer the roles of the players in the Avalon game. "
            "Provide your inferences in the format: 'Player X is likely to be ROLE' for high confidence, "
            "'Player X is suspected to be ROLE' for medium confidence, and 'Player X might be ROLE' for low confidence.\n\n"
            f"Conversation history:\n{response}\n\n"
            "Inferences:"
        )
        model_output = self.generate_response(prompt)
        for line in model_output.split('\n'):
            if "Player" in line and any(confidence in line for confidence in ["is likely to be", "is suspected to be", "might be"]):
                parts = line.split()
                player_id = int(parts[parts.index("Player") + 1])
                if "is likely to be" in line:
                    role = parts[parts.index("be") + 1]
                    confidence = "likely"
                elif "is suspected to be" in line:
                    role = parts[parts.index("be") + 1]
                    confidence = "suspected"
                else:
                    role = parts[parts.index("be") + 1]
                    confidence = "might"
                inferences[player_id] = (role, confidence)
        return inferences

    def filter_memory(self):
        """Filter memory to include the most recent, informative, and reflective messages."""
        recent_messages = self.get_most_recent_messages(10)  # Freshness
        informative_messages = self.get_informative_messages(10)  # Informativeness
        reflection = self.get_reflection_messages()  # Completeness
        filtered_memory = recent_messages + informative_messages + reflection
        
        # Update core memory with inferred roles
        self.update_core_memory_with_summary(filtered_memory)
        
        return self.core_memory + self.rule_memory

    def update_core_memory_with_summary(self, memory):
        """Update core memory with role inferences and generate summary."""
        inferences = self.infer_roles_based_on_context(memory)
        summary = "Based on my memory before, I believe that " + ", ".join([f"Player {player_id} is likely to be {role}" if confidence == "likely" else f"Player {player_id} is suspected to be {role}" if confidence == "suspected" else f"Player {player_id} might be {role}" for player_id, (role, confidence) in inferences.items()])
        self.core_memory = [summary]

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
        """Summarize a given message using the model."""
        inputs = self.tokenizer("Summarize: " + message, return_tensors="pt")
        outputs = self.model.generate(inputs.input_ids, max_length=50, num_return_sequences=1)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

    def retrieve_relevant_messages(self, questions, memory, top_k=5):
        """Retrieve the most relevant messages for each question."""
        encoded_questions = self.sentence_bert_model.encode(questions, convert_to_tensor=True)
        encoded_memory = self.sentence_bert_model.encode(memory, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(encoded_questions, encoded_memory)

        top_k_values, top_k_indices = torch.topk(cosine_scores, k=top_k, dim=1)

        relevant_messages = []
        for i in range(len(questions)):
            relevant_messages.append({
                "question": questions[i],
                "top_k_values": top_k_values[i],
                "top_k_messages": [memory[idx.item()] for idx in top_k_indices[i]]
            })
        
        return relevant_messages

    def get_reflection_messages(self):
        """Generate reflection messages based on the player's role."""
        if self.role == "Merlin":
            predefined_questions = [
                f"As Merlin, how can you subtly guide the good players without revealing your identity?",
                "What information do you have about the identities of the evil players?",
                "How can you help Percival identify you without revealing yourself to the evil players?"
            ]
        elif self.role == "Percival":
            predefined_questions = [
                f"As Percival, how can you distinguish between Merlin and Morgana?",
                "What clues has Merlin given that help you identify him?",
                "How can you protect Merlin while misleading the evil players?"
            ]
        elif self.role == "Loyal Servant":
            predefined_questions = [
                f"As a Loyal Servant, how can you identify and support other good players?",
                "What behavior have you observed that indicates who might be Merlin or evil players?",
                "How can you contribute to the success of the quests?"
            ]
        elif self.role == "Morgana":
            predefined_questions = [
                f"As Morgana, how can you convincingly pretend to be Merlin?",
                "What strategies can you use to mislead Percival and the good players?",
                "How can you coordinate with other evil players to sabotage the quests?"
            ]
        elif self.role == "Assassin":
            predefined_questions = [
                f"As the Assassin, how can you identify who Merlin is?",
                "What behaviors have you observed that indicate Merlin's identity?",
                "How can you balance sabotaging quests and maintaining your cover?"
            ]
        elif self.role == "Oberon":
            predefined_questions = [
                f"As Oberon, how can you act as a good player while causing quests to fail?",
                "What clues can you use to identify your evil teammates?",
                "How can you avoid revealing your identity to Merlin?"
            ]
        
        extra_questions = [f"What do you think about the recent message from other players: '{msg}'?" for msg in self.get_most_recent_messages(7)]
        questions = predefined_questions + extra_questions
        
        relevant_messages = self.retrieve_relevant_messages(questions, self.short_term_memory)
        
        answers = []
        for item in relevant_messages:
            question = item["question"]
            memory = item["top_k_messages"]
            answer_prompt = f"Answer the following question based on the game context and your memory: {question}\nMemory: {memory}"
            answer = self.generate_response(answer_prompt)
            answers.append(answer)
        
        return answers

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

        role_specific_prompt = ""
        
        if self.role == "Merlin":
            role_specific_prompt = (
                "You are Merlin:\n"
                "- You are the leader of good players and must not expose your identity.\n"
                "- You know the identities of the evil players (including Oberon).\n"
                "- Subtly guide your team, especially Percival.\n"
                "- Gain trust to the good players you know.\n"
                "- Avoid behaviors that expose your role: overly accusing, being too helpful.\n"
                "- Goal: Win without revealing your identity."
            )
        elif self.role == "Percival":
            role_specific_prompt = (
                "You are Percival:\n"
                "- You are on the good side and want to win against the evil side.\n"
                "- You know the identities of Merlin and Morgana, but unsure who is who.\n"
                "- Use subtle hints to guide the team and protect Merlin.\n"
                "- Identify the true Merlin and protect her with your speech to win.\n"
                "- Be cautious not to expose Merlin while deciphering true identities.\n"
                "- Goal: Win while safeguarding Merlin."
            )
        elif self.role == "Loyal Servant":
            role_specific_prompt = (
                "You are Loyal Servant of Arthur:\n"
                "- You are on the good side and want to win against the evil side.\n"
                "- You have no special knowledge, rely on discussion and voting.\n"
                "- Find players you can trust on the good side.\n"
                "- Contribute to the success of Quests.\n"
                "- Goal: Win by helping complete Quests and protecting Merlin."
            )
        elif self.role == "Morgana":
            role_specific_prompt = (
                "You are Morgana:\n"
                "- You are on the evil side and want to win against the good side.\n"
                "- Pretend to be Merlin to mislead Percival and the good side.\n"
                "- Deceive players on the good side and gain trust to prevent Quests' success.\n"
                "- Work to prevent Quests' success.\n"
                "- Goal: Confuse and sabotage to win."
            )
        elif self.role == "Assassin":
            role_specific_prompt = (
                "You are Assassin:\n"
                "- You are on the evil side and want to win against the good side.\n"
                "- Discreetly seek Merlin's identity.\n"
                "- Work to prevent Quests' success.\n"
                "- When confident who Merlin is, assassinate.\n"
                "- Goal: Win by Quest failures or assassinating Merlin."
            )
        elif self.role == "Oberon":
            role_specific_prompt = (
                "You are Oberon:\n"
                "- You are on the evil side but you do not know the other evil players.\n"
                "- You must act alone and avoid revealing your role.\n"
                "- Merlin knows you are evil, so be careful.\n"
                "- Your goal is to join teams and cause them to fail, and try to recognize your evil teammates during the game.\n"
                "- Goal: Win by causing Quest failures."
            )

        initial_prompt = (
            f"{game_rule_prompt}\n\n"
            f"{role_specific_prompt}\n\n"
            f"You are Player {self.player_id}, your role is {self.role}."
            f"You can use other identity to protect your self. don't reveal your true role."
        )
        
        self.rule_memory.append(initial_prompt)

    def reflect(self):
        """Conduct a thorough reflection using the recursive contemplation procedure."""
        prompts = [
            f"You are Player {self.player_id} with role {self.role}. Current situation: Reflect on the current game state and guess the roles of other players. "
            f"Analyze other players based on game dialogues with roles: Merlin, Percival, Loyal Servant of Arthur, Morgana, Assassin, Oberon. Morgana, Assassin, and Oberon are evil; others are good.\n"
            f"Consider:\n"
            f"1. Quest Outcomes: Take into account the results of past missions to analyze players' roles.\n"
            f"2. Role List: Remember the possible roles in the game—Merlin, Percival, two Loyal Servants, Morgana, Assassin, and Oberon—and their alignments.\n"
            f"3. Level of Certainty: Use 'Certain' or 'Unknown' to gauge your confidence in your role guesses for each player.\n"
            f"4. Players Disclosing Evil Roles: Be cautious around players who have openly claimed or hinted at being evil roles like Morgana or Assassin.\n"
            f"5. Prior Guesses: Reflect on your earlier estimations of other players' roles, but don't rely solely on them."
            f"Limit your response to 20 words."
        ]
        first_order_response = self.generate_batch_responses(prompts)[0]

        formulation_prompt = (
            f"Respond in two stages: THINK and SPEAK\n"
            f"Based on what you thought: {first_order_response}\n"
            f"In think, internally strategize using history and consider possible deception.\n"
            f"In speak, organize your language based on your contemplation and speak accordingly.\n"
            f"Understand your role's main objective and break it down into chronological sub-goals based on game history. Your thought process should follow these sub-goals for a systematic approach to the main goal.\n"
            f"Limit your response to 20 words."
        )
        formulation_response = self.generate_response_step(formulation_prompt)

        second_order_perspective_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial thoughts. "
            f"Based on what you thought: {formulation_response}\n"
            f"Analyze how your original SPEAK content might be interpreted by other game roles. Reflect on whether it may inadvertently reveal your role-specific clues.\n"
            f"Consider:\n"
            f"1. The perspectives of each game role, including their probable reactions to your SPEAK content.\n"
            f"2. Any unique hints or clues in your original SPEAK that might disclose your role."
            f"Limit your response to 20 words."
        )
        second_order_response = self.generate_response_step(second_order_perspective_prompt)

        refinement_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial thoughts and others' perspective. "
            f"Now refine your thoughts to ensure your role and intentions remain concealed or properly presented.\n"
            f"Based on what you thought: {second_order_response}\n"
            f"Consider:\n"
            f"1. Target Outcome: Aim to achieve the desired result as your role dictates in the game.\n"
            f"2. Role Alignment: Evaluate whether your THINK and SPEAK contents align well with your role {self.role} in the current game state.\n"
            f"3. Strategy Reevaluation: Consider what changes could be made to your THINK and SPEAK contents to improve your chances of winning as {self.role}.\n"
            f"4. Public and Private Content: Remember that THINK contents are private, while SPEAK contents are publicly visible. Strategize accordingly.\n"
            f"Limit your response to 20 words."
        )
        refined_response = self.generate_response_step(refinement_prompt)

        self.short_term_memory = []
        return refined_response

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(inputs.input_ids, max_length=200, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def generate_response_step(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(inputs.input_ids, max_length=200, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def generate_batch_responses(self, prompts):
        responses = []
        for prompt in prompts:
            response = self.generate_response_step(prompt)
            responses.append(response)
        return responses

    def nominate_team(self, team_size):
        prompt = self.team_proposal_prompt(team_size)
        team = [int(x) for x in prompt.split() if x.isdigit()]
        if self.player_id not in team:
            team = team[:team_size-1] + [self.player_id]
        team = list(set(team))
        if len(team) < team_size:
            team += random.sample([i for i in range(1, 8) if i not in team], team_size - len(team))
        return team[:team_size]

    def team_proposal_prompt(self, team_size):
        if self.role in ['Assassin', 'Oberon', 'Morgana']:
            return self.team_proposal_prompt_evil(team_size)
        else:
            return self.team_proposal_prompt_good(team_size)

    def team_proposal_prompt_good(self, team_size):
        base_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: Propose a team of {team_size} players. "
            f"Reflect on game history, received hints, and other players' statements.\n"
            f"Consider:\n"
            f"1. Assess each player's involvement and actions so far.\n"
            f"2. Review how players have contributed to past Quests.\n"
            f"3. Evaluate any behavior that appears either suspicious or trustworthy.\n"
            f"Then present your team. Give reasons based on past events and behaviors. Do not reveal your role or hints. Usually include yourself; if not, clarify.\n"
            f"Remember: Aim for a good team without evil players."
            f"Here's your memory for reference {self.filter_memory()}"
            f"Limit your response to 20 words."
        )

        if self.role == 'Percival':
            specific_prompt = (
                f"- As Percival, use your knowledge to ensure Merlin is included in the team. Avoid choosing Morgana if possible.\n"
                f"- Propose a team that can successfully complete the quest and provide information to identify evil players.\n"
            )
        elif self.role == 'Merlin':
            specific_prompt = (
                f"- As Merlin, subtly guide the proposal to include trustworthy good players and avoid evil players.\n"
                f"- Provide hints to Percival to help him identify you without revealing your identity to others.\n"
            )
        elif self.role == 'Loyal Servant':
            specific_prompt = (
                f"- As a Loyal Servant, choose players who have demonstrated trustworthy behavior in previous rounds.\n"
                f"- Avoid choosing players who have acted suspiciously or contributed to failed quests.\n"
            )

        full_prompt = base_prompt + specific_prompt
        response = self.generate_response(full_prompt)
        return response

    def team_proposal_prompt_evil(self, team_size):
        base_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: Propose a team of {team_size} players. "
            f"Reflect on game history, received hints, and other players' statements.\n"
            f"Consider:\n"
            f"1. Assess each player's involvement and actions so far.\n"
            f"2. Review how players have contributed to past Quests.\n"
            f"3. Evaluate any behavior that appears either suspicious or trustworthy.\n"
            f"Then present your team. Give reasons based on past events and behaviors. Do not reveal your role or hints. Usually include yourself; if not, clarify.\n"
            f"Remember: To win as an evil player, you need to act like a good player. Convince good players you are trustworthy and occasionally include evil players to cause quest failures strategically."
            f"Here's your memory for reference {self.filter_memory()}"
            f"Limit your response to 20 words."
        )

        if self.role == 'Oberon':
            specific_prompt = (
                f"- As Oberon, you do not know the other evil players.\n"
                f"- Your goal is to join teams and cause them to fail.\n"
                f"- Try to act as a good player to gain trust and subtly recognize your evil teammates during the game.\n"
            )
        elif self.role == 'Morgana':
            specific_prompt = (
                f"- As Morgana, act convincingly as Merlin and propose a team that misleads Percival and other good players.\n"
                f"- Include other evil players in a way that seems justified based on their previous actions.\n"
            )
        elif self.role == 'Assassin':
            specific_prompt = (
                f"- As Assassin, propose a team that includes at least one evil player to increase the chances of a quest failure.\n"
                f"- Observe and gather information about other players to help identify Merlin for the endgame.\n"
            )

        full_prompt = base_prompt + specific_prompt
        response = self.generate_response(full_prompt)
        return response

    def vote_for_team_prompt(self, proposed_team):
        reflection_prompt = self.reflect()
        base_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: Vote to approve or reject the proposed team: {proposed_team}.\n"
            f"Based on your reflection: {reflection_prompt}"
            f"Consider:\n"
            f"1. Assess each player's involvement and actions so far.\n"
            f"2. Review how players have contributed to past Quests.\n"
            f"3. Evaluate any behavior that appears either suspicious or trustworthy.\n"
            f"Then clearly state '[approve]' or '[reject]'.\n"
            f"Your final decision will be remembered by other players and yourself for future reasoning."
            f"Here's your memory for reference {self.filter_memory()}"
            f"Limit your response to 10 words."
        )

        if self.role in ['Merlin', 'Percival', 'Loyal Servant']:
            general_refinement_prompt = (
                f"- As a good player, vote based on the trustworthiness of the proposed team members.\n"
                f"- Reject teams that include players who have acted suspiciously or caused quest failures.\n"
            )

            specific_refinement_prompt = ""
            if self.role == 'Percival':
                specific_refinement_prompt = (
                    f"- As Percival, ensure your vote helps identify and protect Merlin. If you suspect Morgana is on the team, vote to reject.\n"
                    f"- Support teams that you believe Merlin would trust.\n"
                )
            elif self.role == 'Merlin':
                specific_refinement_prompt = (
                    f"- As Merlin, your vote should aim to guide good players without revealing your identity.\n"
                    f"- Subtly support teams with trustworthy players and vote against teams with known evil players.\n"
                )
            elif self.role == 'Loyal Servant':
                specific_refinement_prompt = (
                    f"- As a Loyal Servant, vote based on the trustworthiness of the proposed team members.\n"
                    f"- Reject teams that include players who have acted suspiciously or caused quest failures.\n"
                )

            refinement_prompt = general_refinement_prompt + specific_refinement_prompt
        elif self.role in ['Assassin', 'Morgana', 'Oberon']:
            general_refinement_prompt = (
                f"- As an evil player, aim to balance your voting to appear like a good player while strategically causing quest failures.\n"
                f"- Support teams that include other evil players subtly to increase the chances of quest failures.\n"
            )

            specific_refinement_prompt = ""
            if self.role == 'Oberon':
                specific_refinement_prompt = (
                    f"- As Oberon, try to join teams and cause them to fail.\n"
                    f"- Act as a good player to gain trust, and subtly recognize your evil teammates during the game.\n"
                )
            elif self.role == 'Morgana':
                specific_refinement_prompt = (
                    f"- As Morgana, vote to mislead Percival and other good players.\n"
                    f"- Support teams that include evil players while maintaining your disguise as a good player.\n"
                )
            elif self.role == 'Assassin':
                specific_refinement_prompt = (
                    f"- As Assassin, while supporting the evil team, pay extra attention to identifying the real Merlin.\n"
                    f"- Observe players' behaviors and statements to gather clues about Merlin's identity.\n"
                    f"- Balance your statements to maintain your disguise as a good player.\n"
                )

            refinement_prompt = general_refinement_prompt + specific_refinement_prompt

        full_prompt = base_prompt + refinement_prompt
        response = self.generate_response(full_prompt)
        initial_vote = response

        perspective_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on your initial vote: {initial_vote}. "
            f"Consider how other players (both good and evil) might perceive your vote. "
            f"Revise your vote to ensure your role and intentions remain concealed or properly presented."
        )
        other_players_perspective = self.generate_response(perspective_prompt)

        refinement_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on your initial vote and others' perspective: {other_players_perspective}. "
            f"Now refine your vote to ensure your role and intentions remain concealed or properly presented."
            f"Limit your response to state '[approve]' or '[reject]'."
        )
        final_vote = self.generate_response(refinement_prompt)

        return final_vote

    def vote_for_team(self, team):
        prompt = self.vote_for_team_prompt(team)
        return "Approve" if "approve" in prompt.lower() else "Reject"

    def quest_vote_prompt(self):
        reflection_prompt = self.reflect()
        base_prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: Vote success or fail for the Quest. "
            f"Based on your reflection: {reflection_prompt}"
            f"Consider:\n"
            f"1. Assess each player's involvement and actions so far.\n"
            f"2. Review how players have contributed to past Quests.\n"
            f"3. Evaluate any behavior that appears either suspicious or trustworthy.\n"
            f"Then clearly vote [success] or [fail], making your stance unambiguous. Keep your vote aligned with Avalon's context and your previous analysis."
            f"Here's your memory for reference {self.filter_memory()}"
            f"Limit your response to 10 words."
        )

        if self.role == 'Merlin':
            role_specific_prompt = (
                f"- You are Merlin. Vote success to subtly guide good players to trust you.\n"
                f"- Reflect on past quest outcomes and players' behaviors to determine the best course of action.\n"
            )
        elif self.role == 'Percival':
            role_specific_prompt = (
                f"- You are Percival. Vote success to protect Merlin and help the good side.\n"
                f"- Use your knowledge of Merlin's and Morgana's identities to inform your decision.\n"
            )
        elif self.role == 'Loyal Servant':
            role_specific_prompt = (
                f"- You are a Loyal Servant. Vote success to support the good side.\n"
                f"- Reflect on players' contributions to previous quests to make an informed decision.\n"
            )
        elif self.role == 'Assassin':
            role_specific_prompt = (
                f"- You are the Assassin. Vote fail to cause the quest to fail and gather information on Merlin's identity.\n"
                f"- Reflect on players' behaviors and statements to identify Merlin.\n"
            )
        elif self.role == 'Morgana':
            role_specific_prompt = (
                f"- You are Morgana. Vote fail to sabotage the quest and maintain your disguise as a good player.\n"
                f"- Reflect on players' behaviors and statements to identify Percival and mislead him.\n"
            )
        elif self.role == 'Oberon':
            role_specific_prompt = (
                f"- You are Oberon. Vote fail to sabotage the quest while maintaining your cover as a good player.\n"
                f"- Reflect on players' behaviors to identify your evil teammates.\n"
            )

        full_prompt = base_prompt + role_specific_prompt
        response = self.generate_response(full_prompt)
        initial_vote = response

        perspective_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on your initial vote: {initial_vote}. "
            f"Consider how other players (both good and evil) might perceive your vote. "
            f"Revise your vote to ensure your role and intentions remain concealed or properly presented."
        )
        other_players_perspective = self.generate_response(perspective_prompt)

        refinement_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. "
            f"Reflect on your initial vote and others' perspective: {other_players_perspective}. "
            f"Now refine your vote to ensure your role and intentions remain concealed or properly presented."
            f"Limit your response to [success] or [fail]."
        )
        final_vote = self.generate_response(refinement_prompt)

        return final_vote

    def execute_quest(self):
        prompt = self.quest_vote_prompt()
        return "Success" if "success" in prompt.lower() else "Fail"

    def assassinate_merlin_prompt(self):
        reflection_prompt = self.reflect()
        prompt = (
            f"You are Player {self.player_id} with role {self.role}. Current situation: You are evil and The evil team is close to losing; you must guess who Merlin is. "
            f"Consider players' past actions and behaviors to identify Merlin.\n"
            f"Based on your reflection: {reflection_prompt}"
            f"Consider:\n"
            f"1. Assess each player's involvement and actions so far.\n"
            f"2. Review how players have contributed to past Quests.\n"
            f"3. Combine with your previous guesses about other players.\n"
            f"Then state your guess by providing a Player ID, like 'Player 2'."
            f"Here's your memory for reference {self.filter_memory()}"
            f"Limit your response to a Player ID, like 'Player 2'"
        )
        response = self.generate_response(prompt)
        return response

    def debate(self, proposed_team, is_leader=False, previous_statements=[]):
        team_size = len(proposed_team)
        reflection_prompt = self.reflect()
        first_order_prompt = (
            f"You are Player {self.player_id} with role {self.role}. {'As the leader, ' if is_leader else ''}"
            f"Based on your reflection: {reflection_prompt}"
            f"debate about the proposed team of {team_size} players: {proposed_team}. Reflect on the team composition, the leader's reasons for choosing this team, and make inferences about the identities of other players based on the proposed team and their behaviors so far. "
            f"Consider the previous statements: {previous_statements}, this is very important. "
            f"Remember, your identity should not be easily revealed. Evil players should try to gain trust by appearing good, while good players should try to identify other good players. "
            f"Your debate statements will be seen and recorded by all players, so be cautious. "
            f"Here's your memory for reference {self.filter_memory()}. "
            f"Discuss and analyze the previous statements, try to deduce their identity based on their statements. "
            f"Feel free to deceive other players in this gaming environment. "
            f"You may know your goal but all you say is for other players and they will remember it. So, be careful and try to deceive them and hide your goal. "
            f"End your debate by stating your vote (approve or reject) and providing a brief reason (10 words) for your decision. Additionally, give a suggestion to the leader for finalizing the team composition (10 words). This is part of the debate and will be seen by all players."
        )
        first_order_response = self.generate_response_step(first_order_prompt)

        first_order_perspective_prompt = (
            f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial debate thoughts: {first_order_response}. "
            f"Analyze how your original debate content might be interpreted by other game roles. Reflect on whether it may inadvertently reveal your role-specific clues."
        )
        first_order_perspective_response = self.generate_response_step(first_order_perspective_prompt)

        if self.role in ['Merlin', 'Percival', 'Loyal Servant']:
            general_refinement_prompt = (
                f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial debate thoughts and others' perspective: {first_order_perspective_response}. "
                f"Now refine your debate content to ensure your role and intentions remain concealed or properly presented.\n"
                f"Consider the following strategies:\n"
                f"- As a good player, identify and point out evil players who might be in the proposed team.\n"
                f"- Use concrete evidence from past rounds and players' behaviors to support your claims.\n"
                f"- Emphasize the importance of keeping the team free of evil players for the success of the quest.\n"
                f"Limit your response to 20 words."
            )

            specific_refinement_prompt = ""
            if self.role == 'Percival':
                specific_refinement_prompt = (
                    f"- As Percival, ensure you correctly identify Merlin. If mistaken, do not expose Merlin and quickly correct your error.\n"
                    f"- Carefully interpret the information Merlin provides and propose the correct team.\n"
                    f"- Confuse and provoke the evil players during the debate to protect Merlin.\n"
                )
            elif self.role == 'Merlin':
                specific_refinement_prompt = (
                    f"- As Merlin, hide your identity to avoid being discovered by the Assassin.\n"
                    f"- Subtly release information to guide good players, especially Percival, to identify you.\n"
                    f"- Balance your statements to avoid revealing your role while still providing useful hints.\n"
                )
            elif self.role == 'Loyal Servant':
                specific_refinement_prompt = (
                    f"- As a Loyal Servant, avoid hastily accusing certain players to prevent alienation.\n"
                    f"- Use logic to identify inconsistencies in other players' statements and actions.\n"
                    f"- Express your opinions confidently when you spot logical flaws.\n"
                    f"- Act convincingly to confuse the Assassin and increase Merlin's chances of avoiding detection.\n"
                    f"- Be prepared to sacrifice yourself to protect Merlin if necessary.\n"
                )

            refinement_prompt = general_refinement_prompt + specific_refinement_prompt
        elif self.role in ['Assassin', 'Morgana', 'Oberon']:
            general_refinement_prompt = (
                f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial debate thoughts and others' perspective: {first_order_perspective_response}. "
                f"Now refine your debate content to ensure your role and intentions remain concealed or properly presented.\n"
                f"Consider the following strategies:\n"
                f"- As an evil player, emphasize your good intentions and argue convincingly for the approval of the team.\n"
                f"- Try to include other evil players if you are the leader.\n"
                f"- Support other evil players subtly by validating their points and casting doubt on good players.\n"
                f"- Work on identifying Oberon by observing players' behaviors and statements.\n"
                f"- Never expose your identity."
                f"Limit your response to 20 words."
            )

            specific_refinement_prompt = ""
            if self.role == 'Oberon':
                specific_refinement_prompt = (
                    f"- As Oberon, you do not know the other evil players.\n"
                    f"- Your goal is to join teams and cause them to fail.\n"
                    f"- Act as a good player to gain trust, and subtly recognize your evil teammates during the game.\n"
                )
            elif self.role == 'Morgana':
                specific_refinement_prompt = (
                    f"- As Morgana, you should mislead Percival by pretending to be Merlin.\n"
                    f"- Discredit good players by pointing out inconsistencies in their statements or actions.\n"
                    f"- Coordinate subtly with other evil players to strengthen your collective deception.\n"
                )
            elif self.role == 'Assassin':
                specific_refinement_prompt = (
                    f"- As Assassin, while supporting the evil team, pay extra attention to identifying the real Merlin.\n"
                    f"- Observe players' behaviors and statements to gather clues about Merlin's identity.\n"
                    f"- Balance your statements to maintain your disguise as a good player.\n"
                )
            
            refinement_prompt = general_refinement_prompt + specific_refinement_prompt

        refined_debate_response = self.generate_response(refinement_prompt)

        return refined_debate_response

    def finalize_team(self, initial_team, debate_feedback):
        reflection_prompt = self.reflect()
        prompts = [
            f"You are Player {self.player_id} with role {self.role}. Finalize your proposed team after the debate. "
            f"Based on your reflection: {reflection_prompt}"
            f"Initial team: {initial_team}. Consider the feedback received during the debate: {debate_feedback}. "
            f"Your final decision on the team composition will be seen and recorded by all players, so be cautious. "
            f"Here's your memory for reference {self.filter_memory()}"
            f"Generate an initial formulation of your final decision.",

            f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial final decision. "
            f"Analyze how your original final decision might be interpreted by other game roles. Reflect on whether it may inadvertently reveal your role-specific clues.",

            f"You are Player {self.player_id}, your role is {self.role}. Reflect on your initial final decision and others' perspective. "
            f"Now refine your final decision to ensure your role and intentions remain concealed or properly presented."
            f"Limit your response to 20 words"
        ]

        responses = self.generate_batch_responses(prompts)
        final_team = [int(x) for x in responses[-1].split() if x.isdigit()]
        if len(final_team) != len(initial_team):
            final_team = initial_team
        return final_team
