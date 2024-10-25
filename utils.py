import re

def extract_option(s, num):
    # Look for string after [1]: and between "
    match = re.search(r'\[' + str(num) + '\]: "([^"]*)"', s)
    return match.group(1) if match else None

def extract_citation_title(text):
    pattern = r'written the paper with the title "([^"]*)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None


def extract_movie(text):
    marker = "] description: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string

def extract_news_cat(text):
    marker = "] article: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string

def extract_news_headline(text):
    marker = "Generate a headline for the following article: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string

def extract_product_review(text):
    marker = "without further explanation. review: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string


def extract_scholarly_title(text):
    marker = "Generate a title for the following abstract of a paper: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string


def extract_tweet_paraphrasing(text):
    marker = "Paraphrase the following tweet without any explanation before or after it: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string

def get_first_k_tokens(text, k):
    """
    Extracts the first k tokens from a text string.

    :param text: The input text string.
    :param k: The number of tokens to extract.
    :return: The first k tokens of the text string.
    """
    # Split the text into tokens based on whitespace
    tokens = text.split()
    output = " ".join(tokens[:k])

    # Return the first k tokens
    return output

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )




name2taskid = {
    "citation": "LaMP_1",
    "movie_tagging": "LaMP_2M",
    "news_categorize": "LaMP_2N",
    "news_headline": "LaMP_4",
    "product_rating": "LaMP_3",
    "scholarly_title": "LaMP_5",
    "tweet_paraphrase": "LaMP_7"
}