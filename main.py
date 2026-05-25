from generator import generate

EXIT_PHRASES = {
    "quit", "exit", "q",
    "bye", "goodbye",
    "thankyou", "thank you", "thanks",
}


def should_exit(query: str) -> bool:
    normalized = query.lower().strip()
    compact = normalized.replace(" ", "")
    return normalized in EXIT_PHRASES or compact in EXIT_PHRASES


def main():
    print("=" * 50)
    print("  RAG Assistant — ask me anything!")
    print("  Type 'quit', 'bye', or 'thank you' to exit")
    print("=" * 50)

    while True:
        query = input("\nYou: ").strip()

        if not query:
            continue
        if should_exit(query):
            print("Goodbye!")
            break

        print("\nSearching knowledge base...")
        result = generate(query)

        # Show which chunks were retrieved (transparency)
        print("\n--- Retrieved chunks ---")
        for i, chunk in enumerate(result["chunks"], 1):
            print(f"  [{i}] {chunk['title']} (similarity: {chunk['score']})")

        # Show the answer
        print("\n--- Answer ---")
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()
