# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
# model = BertModel.from_pretrained("bert-base-multilingual-uncased")
# text = "I am excited to finish this project"
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
from datasets import load_dataset, DatasetDict
from bert import BertModel
from tokenizer import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

bert_multi = BertModel.from_pretrained("bert-base-multilingual-uncased")
fr_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
# # noneng_sentiment_dataset = load_dataset("tyqiangz/multilingual-sentiments", 'french')
# noneng_sentiment_dataset = load_dataset('CATIE-AQ/french_book_reviews_fr_prompt_sentiment_analysis')
# # print(noneng_sentiment_dataset)
# # print(noneng_sentiment_dataset['train'])
# noneng_paraphrase_dataset = load_dataset('paws-x', 'fr')

# def truncate(example):
#     return {
#         'inputs': " ".join(example['inputs'].split()[:50]),
#         'targets': example['targets']
#     }

# small_sentiment_dataset = DatasetDict(
#     train=noneng_sentiment_dataset['train'].shuffle(seed=1111).select(range(128)).map(truncate),
#     val=noneng_sentiment_dataset['train'].shuffle(seed=1111).select(range(128, 160)).map(truncate),
# )
# # print(small_sentiment_dataset)
# # print(small_sentiment_dataset['train'][:10])

# small_tokenized_dataset = small_sentiment_dataset.map(
#     lambda example: tokenizer(example['inputs'], padding=True, truncation=True),
#     batched=True,
#     batch_size=16
# )

# small_tokenized_dataset = small_tokenized_dataset.remove_columns(["inputs"])
# #small_tokenized_dataset = small_tokenized_dataset.rename_column("label", "labels")
# small_tokenized_dataset.set_format("torch")
# #print(small_tokenized_dataset['train'][0:2])
# print(small_tokenized_dataset)

# train_dataloader = DataLoader(small_tokenized_dataset['train'], batch_size=16)

# for batch in tqdm(train_dataloader):
#     print("worked")
# # print(len(noneng_paraphrase_dataset['validation']))
# # print(len(noneng_paraphrase_dataset['test']))
# print(noneng_paraphrase_dataset)
# print(noneng_paraphrase_dataset['train'][:10])

# noneng_similarity_dataset = load_dataset('CATIE-AQ/stsb_multi_mt_fr_prompt_sentence_similarity')
# print(len(noneng_similarity_dataset['train']))
# print(noneng_similarity_dataset['train'])

def truncateSentencePair_fr(example):
    return {
        'sentence1': " ".join(example['sentence1'].split()[:50]),
        'sentence2': " ".join(example['sentence2'].split()[:50]),
        'label': example['label']
    }

fr_paraphrase_dataset_orig = load_dataset('paws-x', 'fr')

fr_paraphrase_dataset = DatasetDict(
    train=fr_paraphrase_dataset_orig['train'].shuffle(seed=1111).select(range(128)).map(truncateSentencePair_fr),
    val=fr_paraphrase_dataset_orig['validation'].shuffle(seed=1111).select(range(128, 160)).map(truncateSentencePair_fr),
)

print(fr_paraphrase_dataset)
print(fr_paraphrase_dataset['train'][0])
fr_paraphrase_tokenized_dataset = fr_paraphrase_dataset.map(
    lambda example: fr_tokenizer(text=example['sentence1'], text_pair=example['sentence2'], padding=True, truncation=True),
    batched=True,
    batch_size=8
)

print(fr_paraphrase_tokenized_dataset)
print(fr_paraphrase_tokenized_dataset['train']['input_ids'])

