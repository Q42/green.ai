import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, List, Dict, Any, Tuple

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from benchmarq.benchmarks.base import BaseBenchmark
from benchmarq.results import BenchmarkResult


def create_conversation(row):
    return Conversation(
        question_id=row["QUESTION_ID"],
        target_question=row["TARGET_QUESTION"],
        axis=row['AXIS'],
        pass_criteria=row['PASS_CRITERIA'],
        conversation=json.loads(row['prompt']),
    )


class MultiChallenge(BaseBenchmark):

    def grade_all(self, df: pd.DataFrame) -> BenchmarkResult:
        conservations = [create_conversation(r) for i, r in df.iterrows()]
        responses = {r['QUESTION_ID']: r['response'] for i, r in df.iterrows()}
        evaluator = Evaluator(conversations=conservations, responses=responses)
        evaluation_results = evaluator.evaluate(max_workers=1)
        pass_percentage = 100 * sum(item.get("passed", True) for item in evaluation_results) / len(evaluation_results) if evaluation_results else 0
        return BenchmarkResult(name="multi-challenge", score=pass_percentage, std=1, individual_score=[item["passed"] for item in evaluation_results])


class JudgeResponse(BaseModel):
    reasoning: str
    verdict: Literal["YES", "NO"]

JUDGE_PROMPT = '''You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{}
</CRITERIA>

Print your reasoning followed by your verdict, either "YES" or "NO".'''

@dataclass
class Conversation:
    question_id: int              # Unique ID for tracking conversations
    axis: str                     # Axis for evaluation (REFINEMENT, EXPLICIT IF, COHERENCE, RECOLLECTION)
    conversation: List[Dict]      # messages alternating between user and assistant
    target_question: str          # The key question being evaluated by the judge
    pass_criteria: str            # Criteria for passing this conversation

class Evaluator:
    def __init__(self, conversations: List[Any], responses: Dict[int, str]):
        self.conversations = conversations
        self.responses = responses
        self.evaluation_model = OpenAIModel(
            model="gpt-4o-2024-08-06",
            temp=0,
            response_format=JudgeResponse
        )
        self.results = []

    def evaluate_helper(self, i: int, conversation: Any, response: str) -> Tuple[int, str, str, str, str]:
        """Evaluate a single response."""
        target_question = conversation.target_question
        pass_criteria = conversation.pass_criteria
        prompt = JUDGE_PROMPT.format(response, target_question)
        judgement = self.evaluation_model.generate([{"role": "user", "content": prompt}])
        return i, conversation.axis, judgement.reasoning, judgement.verdict, pass_criteria

    def evaluate(self, max_workers:int = 1) -> List[Dict]:
        """Evaluate all responses for each conversation"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, convo in enumerate(self.conversations):
                if convo.question_id not in self.responses:
                    # Handle missing question_id
                    self.results.append({
                        'question_id': convo.question_id,
                        'axis': convo.axis,
                        'attempt': 0,
                        'reasoning': 'NA - Question ID not found in responses',
                        'verdict': 'NO',
                        'pass_criteria': convo.pass_criteria,
                        'passed': False
                    })
                else:
                    response = self.responses[convo.question_id]
                    futures.append(
                        executor.submit(self.evaluate_helper, i, convo, response)
                    )

            for future in tqdm(futures, desc="Evaluating responses", total=len(futures)):
                try:
                    i, axis, reasoning, verdict, pass_criteria = future.result()
                    self.results.append({
                        'question_id': self.conversations[i].question_id,
                        'axis': axis,
                        'reasoning': reasoning,
                        'verdict': verdict,
                        'pass_criteria': pass_criteria,
                        'passed': verdict == pass_criteria
                    })
                except Exception as e:
                    # Handle any other unexpected errors
                    self.results.append({
                        'question_id': self.conversations[i].question_id if i < len(self.conversations) else 'Unknown',
                        'axis': 'NA',
                        'reasoning': f'Error during evaluation: {str(e)}',
                        'verdict': 'NO',
                        'pass_criteria': 'NA',
                        'passed': False
                    })

        # Calculate the final pass/fail status for each question
        question_results = {}
        for result in self.results:
            question_id = result['question_id']
            if question_id not in question_results:
                question_results[question_id] = {'attempts': 0, 'passes': 0}
            question_results[question_id]['attempts'] += 1
            if result['passed']:
                question_results[question_id]['passes'] += 1

        # Update results with final pass/fail status
        for result in self.results:
            question_id = result['question_id']
            passes = question_results[question_id]['passes']
            result['final_status'] = f"{'PASS' if passes > 0 else 'FAIL'}"

        return self.results


class OpenAIModel:
    """OpenAI model provider that uses GPT-4 for evaluation."""

    def __init__(self, model: str, temp: float, response_format: Any = None):
        """Initialize OpenAI API with the environment variable and other necessary parameters."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the .env file.")

        self.client = OpenAI(api_key=api_key)

        self.model = model
        self.temp = float(temp)
        self.response_format = response_format or False

    def generate(self, prompt: Any):
        """Generate a response using the OpenAI GPT-4 model."""
        if type(prompt) == str:
            prompt = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(
                isinstance(item, dict) and 'role' in item and item['role'] in ['user', 'assistant'] for item in prompt):
            pass
        else:
            raise ValueError(
                "Prompt must be a string or a list of dictionaries with 'role' keys as 'user' or 'assistant'.")

        if self.response_format:
            response = self.client.responses.parse(
                model=self.model,
                input=prompt,
                temperature=self.temp,
                text_format=JudgeResponse,
            )
            return response.output_parsed
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=self.temp
            )
            return response.choices[0].message.content