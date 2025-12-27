import time
from typing import List

from swarm_rag.core import Heuristics, SwarmRetriever
from swarm_rag.integrations.stark import StarkInMemoryVectorStore, StarkPreComputedEmbeddingHandler, StarkSKBAdapter
from swarm_rag.eval import Evaluator, EvalReporter
from load_stark import load_and_download_embeddings, load_and_download_skb, load_and_download_qa

# def stark_locality_deposit(ctx):
#     graph: SKB = ctx["graph"]
#     target_id = ctx["target_id"]
#     graph.

# Heuristics.stark_locality_deposit = stark_locality_deposit

def test_first_10_questions(dataset_names: List[str]) -> None:
    """
    Test the first 10 questions of each QA dataset and generate evaluation metrics and plots.
    """
    reporter = EvalReporter()

    for dataset_name in dataset_names:
        print(f"\n{'=' * 50}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'=' * 50}")

        qa_data = load_and_download_qa(dataset_name)
        skb = load_and_download_skb(dataset_name)
        query_embs, doc_embs = load_and_download_embeddings(dataset_name)

        vector_store = StarkInMemoryVectorStore(doc_embs)
        graph_store = StarkSKBAdapter(skb, dataset_name)
        embedding_provider = StarkPreComputedEmbeddingHandler(query_embs)

        retriever = SwarmRetriever(
            vector_store=vector_store,
            graph_store=graph_store,
            embedding_provider=embedding_provider,
            max_workers=8,
        )

        evaluator = Evaluator(k_values=[1, 5, 10, 20], index_name=dataset_name)

        num_questions = min(10, len(qa_data))
        query_results = []

        for i in range(num_questions):
            query, query_id, answer_ids, _ = qa_data[i]

            print(f"\nProcessing question {i + 1}/{num_questions}: {query[:150]}...")

            start_time = time.time()
            retrieved_nodes = retriever.retrieve(
                query=query_id,
                n_agents=20,
                steps=4,
                decay=0.5,
                initial_pool_size=30,
                start_subset=10,
                top_k=20,
                movement_strategies={
                    "semantic": (Heuristics.semantic_similarity, 0.3),
                    "centrality": (graph_store.centrality_heuristic, 0.4),
                    "diversity": (Heuristics.pheromone_repulsion, 0.3),
                },
                deposit_strategies={
                    "collaborative": (
                        Heuristics.deposit_collaborative_amplification,
                        1.0,
                    )
                },
            )
            latency = time.time() - start_time

            metrics = evaluator.calculate_metrics(
                retrieved_nodes=retrieved_nodes,
                ground_truth_ids=answer_ids,
                latency_sec=latency,
            )
            query_results.append(metrics)

            print(
                f"  Hit@1: {metrics['Hit@1']:.3f}, "
                f"Hit@5: {metrics['Hit@5']:.3f}, "
                f"MRR: {metrics['MRR']:.3f}, "
                f"Latency: {latency:.3f}s"
            )

        aggregated = evaluator.aggregate_results(query_results)
        reporter.add_run(dataset_name, aggregated)

        print(f"\nAggregated results for {dataset_name}:")
        print(aggregated.to_string(index=False))

        reporter.plot_metrics(aggregated, dataset_name)

    if len(dataset_names) > 1:
        reporter.plot_comparison(reporter.aggregate(evaluator))


if __name__ == "__main__":
    test_first_10_questions(["prime"])
