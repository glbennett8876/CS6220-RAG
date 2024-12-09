from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class MedicalQA:
    """
    A class to query lightweight medical QA models optimized for local use.
    """
    def __init__(self, model_name: str = "microsoft/biogpt", max_length: int = 512):
        """
        Initialize the MedicalQA system.

        Args:
            model_name (str): Name of the Hugging Face model to use.
            max_length (int): Maximum length of generated or answered text.
        """
        print(f"Loading model: {model_name}...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.qa_pipeline = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer, device=-1
        )
        self.max_length = max_length
        print(f"Model {model_name} loaded successfully!")

    def query(self, question: str, context: str = None) -> str:
        """
        Query the model with a medical question.

        Args:
            question (str): The medical question to query.
            context (str): Additional context or background information.

        Returns:
            str: Model's response.
        """
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:" if context else f"Question: {question}\nAnswer:"
        print("Generating response...")
        response = self.qa_pipeline(
            prompt,
            max_length=self.max_length,
            truncation=True,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
        return response[0]["generated_text"].split("Answer:")[-1].strip()


if __name__ == "__main__":
    # Example usage
    models = [
        "microsoft/biogpt",
        "huawei-noah/TinyBERT_General_4L_312D",    # General-purpose tiny model
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",      # Lightweight conversational model
    ]

    for i, model_name in enumerate(models):
        print(f"### Using model: {model_name} ###")
        medical_qa = MedicalQA(model_name=model_name)
        question = "What are the symptoms of diabetes?"
        context = "Patient is experiencing frequent urination and increased thirst."
        response = medical_qa.query(question, context)
        print(f"Q: {question}\nContext: {context}\nA: {response}\n")
        print(i)
