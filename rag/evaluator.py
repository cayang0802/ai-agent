from __future__ import annotations

import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RAGEvaluatorConfig(BaseModel):
    enabled: bool = False


class RAGEvaluator:
    """Evaluates RAG quality using RAGAS and logs scores to Langfuse.

    Metrics (no ground truth required):
    - Faithfulness:     Answer is grounded in retrieved context.
    - AnswerRelevancy:  Answer addresses the query.
    """

    def __init__(self, langfuse, llm, embeddings) -> None:
        self._langfuse = langfuse
        self._llm = llm
        self._embeddings = embeddings

    def ragas_evaluate(
        self,
        query: str,
        contexts: list[str],
        answer: str,
    ) -> None:
        from ragas import evaluate, EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness, AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        ragas_llm = LangchainLLMWrapper(self._llm)
        ragas_emb = LangchainEmbeddingsWrapper(self._embeddings)

        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=contexts,
            response=answer,
        )
        dataset = EvaluationDataset(samples=[sample])
        metrics = [
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ]

        result = evaluate(dataset, metrics=metrics)
        scores: dict = result.scores[0]
        logger.info("RAGAS raw scores: %s", scores)

        span = self._langfuse.start_span(
            name="rag_eval",
            input={"query": query, "contexts": contexts},
            output={"answer": answer},
        )
        submitted = 0
        for name, value in scores.items():
            if value is None or value != value:  # NaN check
                logger.warning("RAG metric %s skipped (NaN/None)", name)
                continue
            span.score_trace(name=name, value=float(value))
            logger.info("RAG metric %s: %.4f", name, float(value))
            submitted += 1

        span.end()
        self._langfuse.flush()
        logger.info(
            "RAG evaluation complete — %d score(s) sent to Langfuse.",
            submitted,
        )
