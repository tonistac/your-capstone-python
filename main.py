import argparse
import logging

from agent import run_agent
from evals import evaluate_response

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Research assistant agent.")
    parser.add_argument("--query", required=True, help="Question to send to the agent.")
    parser.add_argument(
        "--session",
        default="default",
        help="Session ID for conversation history (default: 'default').",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run judge evaluation on the response after the agent replies.",
    )
    args = parser.parse_args()

    response = run_agent(args.query, args.session)
    print(f"\nResponse:\n{response}")

    if args.eval:
        result = evaluate_response(response)
        print(f"\nEvaluation:\n  Score : {result['score']}/5\n  Reason: {result['reason']}")


if __name__ == "__main__":
    main()
