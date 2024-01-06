from transformers import *
from parrot import Parrot

model_names = [
"tuner007/pegasus_paraphrase",
"Vamsi/T5_Paraphrase_Paws",
"prithivida/parrot_paraphraser_on_T5",
]

model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer =PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")


def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
    """
    Generate paraphrased sentences for a given input sentence using a language model.

    Args:
        model: The language model used for generating paraphrases.
        tokenizer: The tokenizer used for tokenizing the input sentence.
        sentence: The input sentence to be paraphrased.
        num_return_sequences: The number of paraphrased sentences to generate.
        num_beams: The number of beams used during generation.

    Returns:
        A list of paraphrased sentences.
    """
    # Tokenize the input sentence
    inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")

    # Generate paraphrased sequences using the language model
    outputs = model.generate(**inputs, num_beams=num_beams, num_return_sequences=num_return_sequences)

    # Decode the generated sequences into sentences
    paraphrased_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return paraphrased_sentences

# Example usage
sentence = "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences."
get_paraphrased_sentences(model, tokenizer, sentence, num_beams=10, num_return_sequences=10)
get_paraphrased_sentences(model, tokenizer, "To paraphrase a source, you have to rewrite a passage without changing the meaning of the original text.", num_beams=10, num_return_sequences=10)
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
get_paraphrased_sentences(model, tokenizer, "paraphrase: " + "One of the best ways to learn is to teach what you've already learned")



parrot = Parrot()
phrases = [ sentence,
"One of the best ways to learn is to teach what you've already learned",
"Paraphrasing is the process of coming up with someone else's ideas in your own words"
]
for phrase in phrases:
    print("-"*100)
    print("Input_phrase: ", phrase)
    print("-"*100)
    paraphrases = parrot.augment(input_phrase=phrase)
    if paraphrases:
        for paraphrase in paraphrases:
            print(paraphrase)
