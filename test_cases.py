"""Tests. Run: python3 test_cases.py"""
import time
import rag

ANSWERABLE = [
    "What is Foundry Local?",
    "How does the RAG pattern work?",
    "Why is SQLite used for local storage?",
    "What is cosine similarity?",
    "What chunk size should I use?",
    "Which instructions matter most in the system prompt?",
]
UNANSWERABLE = [
    "Who won the World Cup in 2018?",
    "What is the capital of Australia?",
    "",
]

def run():
    passed = failed = 0
    for q in ANSWERABLE + UNANSWERABLE:
        expect_answer = q in ANSWERABLE
        started = time.time()
        result = rag.answer_query(q or " ")
        elapsed = time.time() - started
        refused = result["answer"].startswith(rag.NO_ANSWER[:20])
        ok = (not refused) if expect_answer else refused
        passed, failed = (passed + ok, failed + (not ok))
        print("[{}] {:<50} {:.1f}s".format("PASS" if ok else "FAIL", q or "(empty query)", elapsed))
        print("     -> {}".format(result["answer"][:120].replace("\n", " ")))
        if result["sources"]:
            print("     sources: {}".format(", ".join(
                "{} ({:.3f})".format(s["source"], s["score"]) for s in result["sources"])))
        print()
    print("{} passed, {} failed".format(passed, failed))

if __name__ == "__main__":
    run()
