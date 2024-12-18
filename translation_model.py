import os
import sys
import transformers
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import AdamWeightDecay
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"

"""## Helsinki-NLP/opus-mt-en-hi model

source: https://huggingface.co/Helsinki-NLP/opus-mt-en-hi

# The Dataset

Source: https://huggingface.co/datasets/cfilt/iitb-english-hindi
"""

raw_datasets = load_dataset("cfilt/iitb-english-hindi")

raw_datasets

raw_datasets['train'][1]

"""#Preprocessing the data"""

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenizer("Hello, this is a sentence!")

tokenizer(["Hello, this is a sentence!", "This is another sentence."])

# Use the text_target argument instead of as_target_tokenizer
output = tokenizer(["एक्सेर्साइसर पहुंचनीयता अन्वेषक"], text_target=["एक्सेर्साइसर पहुंचनीयता अन्वेषक"])
print(output)

max_input_length = 128
max_target_length = 128

source_lang = "en"
target_lang = "hi"

def preprocess_function(examples):
    # Extract inputs and targets from the translation examples
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]

    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Tokenize targets using `text_target`
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, text_target=targets)

    # Add tokenized labels to model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

preprocess_function(raw_datasets["train"][:2])

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 16
learning_rate = 2e-5
weight_decay = 0.01
num_train_epochs = 1

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")

generation_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=128)

train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
)

validation_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator,
)

generation_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    batch_size=8,
    shuffle=False,
    collate_fn=generation_data_collator,
)

optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)
model.compile(optimizer=optimizer)

# history = model.fit(
#     train_dataset,
#     validation_data=validation_dataset,
#     epochs=50
# )

history=model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

model.save_pretrained("hindi1/")
tokenizer.save_pretrained("hindi1/")

# """# Model Testing"""

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForSeq2SeqLM.from_pretrained("hindi2/")

def load_translation_model():
    """
    Load the pre-trained translation model and tokenizer from local directory.
    Returns:
        model: The translation model.
        tokenizer: The tokenizer for the model.
    """
    model_checkpoint = "hindi2/"  # Directory where the locally trained model is saved
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    return model, tokenizer

def translate_text(text, model, tokenizer):
    """
    Translate text from English to Hindi using the model and tokenizer.
    Args:
        text (str): The input text in English.
        model: The translation model.
        tokenizer: The tokenizer for the model.
    Returns:
        str: Translated text in Hindi.
    """
    # Tokenize the input text
    tokenized = tokenizer([text], return_tensors="np")

    # Generate translation
    translated_tokens = model.generate(**tokenized, max_length=128)

    # Decode the translated tokens to get the Hindi text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text


# text="Who are you?"

# tokenized = tokenizer([text], return_tensors="np")

#     # Generate translation
# translated_tokens = model.generate(**tokenized, max_length=128)
# out = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
# print(out)


import matplotlib.pyplot as plt
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Plot training and validation loss
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()