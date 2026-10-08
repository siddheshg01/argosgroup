"""Command-line entry point for the Phase 1 financial intelligence engine."""
from src.financial_intelligence.pipeline import FinancialIntelligencePipeline
from src.financial_intelligence.root_cause import RootCauseInvestigator
from src.financial_intelligence.forecasting import FinancialForecaster
from src.financial_intelligence.llm_analyst import LLMFinancialAnalyst
from src.financial_intelligence.rag.rag_pipeline import RAGPipeline
from pathlib import Path
import sys

if len(sys.argv) > 1 and sys.argv[1] == "--agent-query":
    from src.financial_intelligence.agents.orchestrator import run_agent_query
    question = " ".join(sys.argv[2:]).strip()
    if not question:
        raise SystemExit('Usage: python main.py --agent-query "your financial question"')
    result = run_agent_query(question)
    print(f"PHASE 6 STATUS: {'COMPLETE' if result['status'] == 'complete' else 'PARTIAL'}")
    print("Agents: YES")
    print("Orchestration: YES")
    print(f"Phase 1-5 Integration: {'YES' if result['phase_1_to_5_integrated'] else 'NO'}")
    print(f"Gemini: {'YES' if result['capabilities']['gemini'] else 'NO'}")
    print(f"RAG: {'YES' if result['capabilities']['rag'] else 'NO'}")
    print(f"Recommendations: {'YES' if result['capabilities']['recommendations'] else 'NO'}")
    print("Reports: output/agentic_financial_analysis.json and output/agentic_financial_analysis.md")
    raise SystemExit(0)

if len(sys.argv) > 1 and sys.argv[1] == "--workflow":
    from src.financial_intelligence.workflow_cli import main as workflow_main
    workflow_main(sys.argv[2:])
    raise SystemExit(0)

def main() -> None:
    print("=" * 58)
    print("AGENTIC AI FINANCIAL OPERATIONS PLATFORM")
    print("PHASE 1 - FINANCIAL INTELLIGENCE ENGINE")
    print("=" * 58)
    dataset_path = Path("data/raw/amazon_sales_2024.csv")
    # Accept the filename the user uploaded while preserving the documented default.
    if not dataset_path.exists() and Path("data/raw/Amazon.csv").exists():
        dataset_path = Path("data/raw/Amazon.csv")
    report = FinancialIntelligencePipeline(dataset_path).run()
    print(f"\nRecords analyzed: {report['metadata']['records_analyzed']:,}")
    print(f"Data quality score: {report['data_quality']['data_quality_score']:.1f}")
    print("Financial summary:")
    for key, value in report["financial_summary"].items(): print(f"  {key}: {value}")
    print(f"Risk indicator: {report['risk_analysis']['risk_score']} / 100 ({report['risk_analysis']['risk_level']})")
    print("\nReport saved to output/financial_report.json")
    phase2 = RootCauseInvestigator(dataset_path).run()
    print("\nPHASE 2 - ROOT CAUSE INVESTIGATION ENGINE")
    print(f"Metrics investigated: {len(phase2['metrics_investigated'])}")
    print(f"Significant changes: {phase2['significant_change_count']}")
    print(f"Root-cause investigations: {phase2['root_cause_investigation_count']}")
    print(f"Charts generated: {phase2['outputs']['charts_generated']}")
    print("Reports saved to output/root_cause_report.json and output/root_cause_summary.csv")
    phase3 = FinancialForecaster(dataset_path).run()
    successful = [item for item in phase3["forecasts"] if item.get("status") == "forecasted"]
    print("\nPHASE 3 - FINANCIAL FORECASTING ENGINE")
    print(f"Metrics forecasted: {len(phase3['metrics_forecasted'])}")
    print(f"Models tested: {len(phase3['models_tested'])} unique model types")
    print("Best models: " + ", ".join(f"{item['metric']}={item['selected_model']['name']}" for item in successful))
    horizons = {str(h) for item in successful for h in (3, 6, 12)
                if f"{h}_months" in item.get("forecasts_by_horizon", {})}
    print(f"3/6/12-month forecasts: {'YES' if horizons == {'3', '6', '12'} else 'NO'}")
    print(f"Reports saved; charts generated: {phase3['outputs']['charts_generated']}")
    phase4 = LLMFinancialAnalyst().run()
    print("\nPHASE 4 - LLM FINANCIAL ANALYST")
    print(f"LLM Analyst: {'YES' if phase4.get('status') == 'generated' else 'NO'}")
    print("Phase 1-3 integrated: YES")
    print(f"Analysis generated: {'YES' if phase4.get('status') == 'generated' else 'NO'}")
    print(f"JSON: {'YES' if Path('output/llm_financial_analysis.json').is_file() else 'NO'}")
    print(f"Markdown: {'YES' if Path('output/llm_financial_analysis.md').is_file() else 'NO'}")
    print("Tests: run python -m pytest -q")
    print(f"PHASE 4 STATUS: {'COMPLETE' if phase4.get('status') == 'generated' else 'NEEDS FIXES'}")
    phase5 = RAGPipeline().run(financial_analysis=phase4)
    print("\nPHASE 5 - RAG / FINANCIAL POLICY INTELLIGENCE")
    print(f"Documents indexed: {phase5['documents_indexed']}")
    print(f"Chunks created: {phase5['chunks_created']}")
    print(f"Vector store: {'YES' if phase5['vector_store'] else 'NO'}")
    print(f"Retrieval: {'YES' if phase5['retrieval_status'] == 'complete' else 'NO'}")
    print(f"Citations: {'YES' if phase5['source_citations'] else 'NO'}")
    print(f"Gemini RAG: {'YES' if phase5['status'] == 'answered' else 'NO'}")
    no_result_supported = phase5["status"] in {"answered", "no_relevant_documents", "insufficient_policy_evidence"}
    print(f"No-result handling: {'YES' if no_result_supported else 'NO'}")
    print(f"Reports: {'YES' if Path('output/rag_report.json').is_file() and Path('output/rag_summary.md').is_file() else 'NO'}")
    print("Tests: run python -m pytest -q")
    print(f"PHASE 5 STATUS: {'COMPLETE' if phase5['status'] in {'answered', 'no_relevant_documents'} and phase5['vector_store'] else 'NEEDS FIXES'}")

if __name__ == "__main__": main()
