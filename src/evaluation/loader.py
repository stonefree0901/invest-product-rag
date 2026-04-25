"""Evaluation set loader for RAG system."""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvaluationLoader:
    """Load and manage evaluation questions."""

    def __init__(self, eval_dir: Path):
        """
        Initialize evaluation loader.

        Args:
            eval_dir: Directory containing evaluation files.
        """
        self.eval_dir = Path(eval_dir)
        self.questions = {}

    def load_questions(self, filename: str) -> List[Dict]:
        """
        Load questions from a JSONL file.

        Args:
            filename: Name of the JSONL file.

        Returns:
            List of question dictionaries.
        """
        file_path = self.eval_dir / filename

        if not file_path.exists():
            logger.error(f"Evaluation file not found: {file_path}")
            return []

        questions = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                questions.append(json.loads(line))

        logger.info(f"Loaded {len(questions)} questions from {filename}")
        return questions

    def filter_by_company(self, questions: List[Dict], company: str) -> List[Dict]:
        """
        Filter questions by company.

        Args:
            questions: List of question dictionaries.
            company: Company name to filter by.

        Returns:
            Filtered list of questions.
        """
        filtered = [q for q in questions if q.get('company') == company]
        logger.info(f"Filtered to {len(filtered)} questions for company: {company}")
        return filtered

    def filter_by_relevance_type(self, questions: List[Dict], relevance_type: str) -> List[Dict]:
        """
        Filter questions by relevance type.

        Args:
            questions: List of question dictionaries.
            relevance_type: Relevance type to filter by (dense/hybrid/sparse).

        Returns:
            Filtered list of questions.
        """
        filtered = [q for q in questions if q.get('relevance_type') == relevance_type]
        logger.info(f"Filtered to {len(filtered)} questions for relevance type: {relevance_type}")
        return filtered

    def filter_by_difficulty(self, questions: List[Dict], difficulty: str) -> List[Dict]:
        """
        Filter questions by difficulty.

        Args:
            questions: List of question dictionaries.
            difficulty: Difficulty level to filter by.

        Returns:
            Filtered list of questions.
        """
        filtered = [q for q in questions if q.get('difficulty') == difficulty]
        logger.info(f"Filtered to {len(filtered)} questions for difficulty: {difficulty}")
        return filtered

    def get_questions_by_id(self, questions: List[Dict], question_ids: List[str]) -> List[Dict]:
        """
        Get specific questions by IDs.

        Args:
            questions: List of question dictionaries.
            question_ids: List of question IDs to retrieve.

        Returns:
            List of matching questions.
        """
        id_set = set(question_ids)
        filtered = [q for q in questions if q.get('question_id') in id_set]
        logger.info(f"Retrieved {len(filtered)} questions by ID")
        return filtered

    def print_summary(self, questions: List[Dict]):
        """
        Print summary statistics for a set of questions.

        Args:
            questions: List of question dictionaries.
        """
        if not questions:
            print("No questions to summarize.")
            return

        print("\n" + "=" * 60)
        print("EVALUATION SET SUMMARY")
        print("=" * 60)
        print(f"Total Questions: {len(questions)}")

        # By company
        companies = {}
        for q in questions:
            company = q.get('company', 'unknown')
            companies[company] = companies.get(company, 0) + 1
        print("\nBy Company:")
        for company, count in sorted(companies.items()):
            print(f"  {company}: {count}")

        # By relevance type
        relevance_types = {}
        for q in questions:
            rtype = q.get('relevance_type', 'unknown')
            relevance_types[rtype] = relevance_types.get(rtype, 0) + 1
        print("\nBy Relevance Type:")
        for rtype, count in sorted(relevance_types.items()):
            print(f"  {rtype}: {count}")

        # By difficulty
        difficulties = {}
        for q in questions:
            diff = q.get('difficulty', 'unknown')
            difficulties[diff] = difficulties.get(diff, 0) + 1
        print("\nBy Difficulty:")
        for diff, count in sorted(difficulties.items()):
            print(f"  {diff}: {count}")

        # Exact match requirement
        exact_match = sum(1 for q in questions if q.get('requires_exact_match'))
        print(f"\nRequires Exact Match: {exact_match} / {len(questions)}")
        print("=" * 60)


def main():
    """Example usage."""
    eval_dir = Path("data/evaluation")
    loader = EvaluationLoader(eval_dir)

    # Load basic questions
    questions = loader.load_questions("basic_questions.jsonl")
    loader.print_summary(questions)

    # Example filtering
    print("\n\n=== Example: Barclays Questions ===")
    barclays_questions = loader.filter_by_company(questions, "barclays")
    for q in barclays_questions:
        print(f"  {q['question_id']}: {q['question']}")

    print("\n\n=== Example: Hybrid Relevance Questions ===")
    hybrid_questions = loader.filter_by_relevance_type(questions, "hybrid")
    for q in hybrid_questions:
        print(f"  {q['question_id']}: {q['question']}")

    return loader


if __name__ == "__main__":
    loader = main()
