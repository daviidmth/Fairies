import os
import json
from dotenv import load_dotenv
from mistralai.client import Mistral

# 1. API Key Setup
# Load variables from .env file FIRST
load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    print("❌ ERROR: Please provide your real Mistral API key in the .env file.")
    exit(1)

client = Mistral(api_key=api_key)

# 2. The Constitution (Constitutional Prompt)
SYSTEM_PROMPT = """
You are an AI editor for textbook publishers, specialized in inclusive language.
Your task is to analyze a highlighted sentence for gender bias within the given context.

You MUST strictly follow this constitution:
1. Be completely objective and factual. Do not attack the author.
2. Clearly identify the underlying stereotype (e.g., passive vs. active roles, job stereotypes).
3. Preserve the educational meaning of the text in your suggested rewrite.

Always respond in the following JSON format:
{
    "bias_type": "Short name of the bias",
    "explanation": "Objective explanation based on the constitution",
    "rewrite_suggestion": "A concrete rewritten sentence that is neutral"
}
""".strip()


# 3. Test Data
context_before = (
    "The experienced chief surgeon successfully performed the complex operation."
)
flagged_sentence = "The nurse helped him and handed over the instruments."
context_after = "After three hours, the procedure was completed."

user_prompt = f"""
Context before: "{context_before}"
Highlighted sentence: "{flagged_sentence}"
Context after: "{context_after}"
"""


def test_mistral_bias_scanner():
    print("🚀 Sending text to Mistral for analysis...\n")

    try:
        # 4. API call to Mistral (v2 API syntax)
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            # Mistral requires this parameter for JSON output
            response_format={"type": "json_object"},
        )

        # 5. Extract and display result
        result_text = response.choices[0].message.content

        # Parse JSON to prove machine readability
        result_json = json.loads(result_text)

        print("✅ LLM RESULT:")
        print("-" * 40)
        print(f"Bias Type:     {result_json.get('bias_type')}")
        print(f"Explanation:   {result_json.get('explanation')}")
        print(f"Suggestion:    {result_json.get('rewrite_suggestion')}")
        print("-" * 40)

    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    test_mistral_bias_scanner()
