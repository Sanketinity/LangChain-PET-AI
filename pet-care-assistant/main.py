# main.py

from src.agent.graph import app

if __name__ == "__main__":
    print("Pet Care RAG Bot is ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        inputs = {"question": user_input}
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Node '{key}':")
        print("------\nFinal Answer:\n------")
        print(output[key]['generation'])
        print("\n")