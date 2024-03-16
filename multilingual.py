'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets_local import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask

from datasets import load_dataset, DatasetDict
from tokenizer import BertTokenizer

TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert_en = BertModel.from_pretrained('bert-base-uncased')
        self.bert_fr = BertModel.from_pretrained('bert-base-multilingual-uncased')
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert_en.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        for param in self.bert_fr.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        self.linear_sentiment_en = torch.nn.Linear(config.hidden_size, len(config.num_labels))
        self.linear_sentiment_fr = torch.nn.Linear(config.hidden_size, 2)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        #Works for  paraphrase detection (double the hidden size since we cat two examples outputs, and out values either 0 or 1 on whether paraphrase)
        #same for french paraphrase (is or isn't)
        self.linear_paraphrase = torch.nn.Linear(2 * config.hidden_size, 1)
        #Works for similarity prediction detection (out values 0 to 5 on how similar)
        #same for french similarity (on a scale of 0 to 1 instead but still a single ouput logit)
        self.linear_similarity = torch.nn.Linear(2 * config.hidden_size, 1)
        #raise NotImplementedError


    def forward(self, input_ids, attention_mask, lang):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        if lang == 'en':
            minbert_out = self.bert_en(input_ids, attention_mask)['pooler_output']
        else:
            minbert_out = self.bert_fr(input_ids, attention_mask)['pooler_output']
        return minbert_out


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        #Initial baseline- just use existing BERT embeddings to attempt downstream tasks
        minbert_out = self.forward(input_ids, attention_mask, lang='en')
        dropout_out = self.dropout(minbert_out)
        logits = self.linear_sentiment_en(dropout_out)
        return logits

    def predict_sentiment_fr(self, input_ids, attention_mask):
        minbert_out = self.forward(input_ids, attention_mask, lang='fr')
        dropout_out = self.dropout(minbert_out)
        logits = self.linear_sentiment_fr(dropout_out)
        return logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        minbert_out_1 = self.forward(input_ids_1, attention_mask_1, lang='en')
        dropout_out_1 = self.dropout(minbert_out_1)
        minbert_out_2 = self.forward(input_ids_2, attention_mask_2, lang='en')
        dropout_out_2 = self.dropout(minbert_out_2)
        combined_minbert= torch.cat((dropout_out_1, dropout_out_2), dim=1)
        logits = self.linear_paraphrase(combined_minbert)
        # combined_input = torch.cat((input_ids_1, input_ids_2), dim=1)
        # combined_attention_mask = torch.cat((attention_mask_1, attention_mask_2), dim=1)
        # minbert_out = self.forward(combined_input, combined_attention_mask) 
        # dropout_out = self.dropout(minbert_out)
        # logits = self.linear_paraphrase(dropout_out)
        # print("minbert_out_1 len: " + str(minbert_out_1.size))
        # print("minbert_out_2 len: " + str(minbert_out_2))
        #print("logits: " + str(logits.shape))
        # print("len logits: " + str(len(logits)))
        #print("combined: " + str(combined_minbert.shape))
        #print("len combined: " + str(len(combined_minbert)))
        return logits


    def predict_paraphrase_fr(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        minbert_out_1 = self.forward(input_ids_1, attention_mask_1, lang='fr')
        dropout_out_1 = self.dropout(minbert_out_1)
        minbert_out_2 = self.forward(input_ids_2, attention_mask_2, lang='fr')
        dropout_out_2 = self.dropout(minbert_out_2)
        combined_minbert= torch.cat((dropout_out_1, dropout_out_2), dim=1)
        logits = self.linear_paraphrase(combined_minbert)
        return logits

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        minbert_out_1 = self.forward(input_ids_1, attention_mask_1, lang='en')
        dropout_out_1 = self.dropout(minbert_out_1)
        minbert_out_2 = self.forward(input_ids_2, attention_mask_2, lang='en')
        dropout_out_2 = self.dropout(minbert_out_2)
        combined_minbert= torch.cat((dropout_out_1, dropout_out_2), dim=1)
        logits = self.linear_similarity(combined_minbert)
        # combined_input = torch.cat((input_ids_1, input_ids_2), dim=1)
        # combined_attention_mask = torch.cat((attention_mask_1, attention_mask_2), dim=1)
        # minbert_out = self.forward(combined_input, combined_attention_mask) 
        # dropout_out = self.dropout(minbert_out)
        # logits = self.linear_similarity(dropout_out)
        return logits
    
    def predict_similarity_fr(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        minbert_out_1 = self.forward(input_ids_1, attention_mask_1, lang='fr')
        dropout_out_1 = self.dropout(minbert_out_1)
        minbert_out_2 = self.forward(input_ids_2, attention_mask_2, lang='fr')
        dropout_out_2 = self.dropout(minbert_out_2)
        combined_minbert= torch.cat((dropout_out_1, dropout_out_2), dim=1)
        logits = self.linear_similarity(combined_minbert)
        # combined_input = torch.cat((input_ids_1, input_ids_2), dim=1)
        # combined_attention_mask = torch.cat((attention_mask_1, attention_mask_2), dim=1)
        # minbert_out = self.forward(combined_input, combined_attention_mask) 
        # dropout_out = self.dropout(minbert_out)
        # logits = self.linear_similarity(dropout_out)
        return logits




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

#My train loading helper function for loadingParaphrase Quora dataset for training
# def load_quora(para_train_data):
#     para_train_data = SentencePairTestDataset(para_test_data, args)
#     para_dev_data = SentencePairDataset(para_dev_data, args)

#     para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
#                                           collate_fn=para_test_data.collate_fn)
#     para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
#                                          collate_fn=para_dev_data.collate_fn)

# #My train loading helper function for loadingParaphrase Quora dataset for training
# def load_semEval():
#     pass
#Defining as a global variable
fr_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

def truncate(example):
    return {
        'inputs': " ".join(example['inputs'].split()[:50]),
        'targets': example['targets']
    }

def truncateSentencePair_fr(example):
    return {
        'sentence1': " ".join(example['sentence1'].split()[:50]),
        'sentence2': " ".join(example['sentence2'].split()[:50]),
        'label': example['label']
    }

def truncateSimilarity_fr(example):
    combined_phrases = example['inputs']
    phrase1_startidx = combined_phrases.find('Phrase 1 :')
    phrase2_startidx = combined_phrases.find('Phrase 2 :')
    #+10 scoots it up to actual sentence start
    phrase1 = combined_phrases[phrase1_startidx + 10:phrase2_startidx]
    phrase2= combined_phrases[phrase2_startidx + 10:]
    return {
        'sentence1': " ".join(phrase1.split()[:50]),
        'sentence2': " ".join(phrase2.split()[:50]),
        'label': example['targets']
    }

def fr_tokenize_helper(example):
    encoding1 = fr_tokenizer(example['sentence1'], return_tensors='pt', padding=True, truncation=True)
    encoding2 = fr_tokenizer(example['sentence2'], return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding1['input_ids'])
    attention_mask = torch.LongTensor(encoding1['attention_mask'])
    token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

    token_ids2 = torch.LongTensor(encoding2['input_ids'])
    attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
    token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])

    # if self.isRegression:
    #     labels = torch.DoubleTensor(labels)
    # else:
    #     labels = torch.LongTensor(labels)

    return {'token_ids_1': token_ids, 
            'token_type_ids_1': token_type_ids,
            'attention_mask_1': attention_mask,
            'token_ids_2': token_ids2, 
            'token_type_ids_2': token_type_ids2, 
            'attention_mask_2': attention_mask2
            }
    #train = example['train']

def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    #sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train, args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    fr_sentiment_dataset_orig = load_dataset('CATIE-AQ/french_book_reviews_fr_prompt_sentiment_analysis')

    fr_sentiment_dataset = DatasetDict(
        train=fr_sentiment_dataset_orig['train'].shuffle(seed=1111).select(range(128)).map(truncate),
        val=fr_sentiment_dataset_orig['train'].shuffle(seed=1111).select(range(128, 160)).map(truncate),
    )

    fr_sentiment_tokenized_dataset = fr_sentiment_dataset.map(
        lambda example: fr_tokenizer(example['inputs'], padding=True, truncation=True),
        batched=True,
        batch_size=8
    )

    fr_sentiment_tokenized_dataset = fr_sentiment_tokenized_dataset.remove_columns(["inputs"])
    #small_tokenized_dataset = small_tokenized_dataset.rename_column("label", "labels")
    fr_sentiment_tokenized_dataset.set_format("torch")
    #print(small_tokenized_dataset['train'][0:2])
    fr_sentiment_train_dataloader = DataLoader(fr_sentiment_tokenized_dataset['train'], batch_size=8)
    fr_sentiment_train_dataloader = DataLoader(fr_sentiment_tokenized_dataset['val'], batch_size=8)
    
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    ######MY CODE#####
    # self.load_quora(para_train_data)

    # self.load_semEval()
    #Paraphrase data prep
    #English:
    #para_train_data = para_train_data[:200]
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)
    
    #French paraphrase
    fr_paraphrase_dataset_orig = load_dataset('paws-x', 'fr')

    fr_paraphrase_dataset = DatasetDict(
        train=fr_paraphrase_dataset_orig['train'].shuffle(seed=1111).select(range(128)).map(truncateSentencePair_fr),
        val=fr_paraphrase_dataset_orig['validation'].shuffle(seed=1111).select(range(128, 160)).map(truncateSentencePair_fr),
    )

    fr_paraphrase_tokenized_dataset = fr_paraphrase_dataset.map(fr_tokenize_helper, batched=True,
        batch_size=8)

    fr_paraphrase_tokenized_dataset = fr_paraphrase_tokenized_dataset.remove_columns(["sentence1"])
    fr_paraphrase_tokenized_dataset = fr_paraphrase_tokenized_dataset.remove_columns(["sentence2"])
    fr_paraphrase_tokenized_dataset = fr_paraphrase_tokenized_dataset.rename_column("label", "labels")
    fr_paraphrase_tokenized_dataset.set_format("torch")
    #print(small_tokenized_dataset['train'][0:2])
    fr_paraphrase_train_dataloader = DataLoader(fr_paraphrase_tokenized_dataset['train'], batch_size=8)
    fr_paraphrase_dev_dataloader = DataLoader(fr_paraphrase_tokenized_dataset['val'], batch_size=8)

    #semantic textual simlarity data prep
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)
    
    #French semantic textual similarity
    fr_similarity_dataset_orig = load_dataset('CATIE-AQ/stsb_multi_mt_fr_prompt_sentence_similarity')

    fr_similarity_dataset = DatasetDict(
        train=fr_similarity_dataset_orig['train'].shuffle(seed=1111).select(range(128)).map(truncateSimilarity_fr),
        val=fr_similarity_dataset_orig['validation'].shuffle(seed=1111).select(range(128, 160)).map(truncateSimilarity_fr),
    )

    #Since I just reformatted the datset to have a column for each phrase and a label, don't need the original setup
    fr_similarity_dataset = fr_similarity_dataset.remove_columns(["inputs"])
    fr_similarity_dataset = fr_similarity_dataset.remove_columns(["targets"])

    fr_similarity_tokenized_dataset = fr_similarity_dataset.map(fr_tokenize_helper, batched=True,
        batch_size=8)

    fr_similarity_tokenized_dataset.set_format("torch")
    fr_similarity_train_dataloader = DataLoader(fr_paraphrase_tokenized_dataset['train'], batch_size=8)
    fr_similarity_dev_dataloader = DataLoader(fr_paraphrase_tokenized_dataset['val'], batch_size=8)
    ####END MY CODE #########
    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
    #     #SENTIMENT LOOPS
    #     #English sentiment:
        for batch in tqdm(sst_train_dataloader, desc=f'trainSentimentEn-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment_en(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            
        #French sentiment:
        label_mapping = {'neg': 0, 'pos': 1}
        for batch in tqdm(fr_sentiment_train_dataloader, desc=f'trainSentimentFr-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['input_ids'],
                                       batch['attention_mask'], batch['targets'])
    
            #Change pos and neg to 1 and 0, and make it a long tensor instead of a list for cross entropy
            converted_labels = [label_mapping[label] for label in b_labels]
            b_labels = torch.LongTensor(converted_labels)

            optimizer.zero_grad()
            logits = model.predict_sentiment_fr(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels, reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1


        # #Train a batch for paraphrase
        #English paraphrase:
        for batch in tqdm(para_train_dataloader, desc=f'trainParaEn-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_type_ids_1, b_mask_1, b_ids_2, b_type_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                       batch['token_type_ids_1'], batch['attention_mask_1'], batch['token_ids_2'],
                                       batch['token_type_ids_2'], batch['attention_mask_2'], batch['labels'])
            b_ids_1 = b_ids_1.to(device)
            b_type_ids_1 = b_type_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_type_ids_2 = b_type_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            loss = F.cross_entropy(logits.float().squeeze(), b_labels.float().view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        #French paraphrase:
        for batch in tqdm(fr_paraphrase_train_dataloader, desc=f'trainParaFr-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_type_ids_1, b_mask_1, b_ids_2, b_type_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                       batch['token_type_ids_1'], batch['attention_mask_1'], batch['token_ids_2'],
                                       batch['token_type_ids_2'], batch['attention_mask_2'], batch['labels'])
            b_ids_1 = b_ids_1.to(device)
            b_type_ids_1 = b_type_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_type_ids_2 = b_type_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_paraphrase_fr(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            loss = F.cross_entropy(logits.float().squeeze(), b_labels.float().view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        #Train a batch for semantic similarity
        #English similarity:
        for batch in tqdm(sts_train_dataloader, desc=f'trainSimEn-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_type_ids_1, b_mask_1, b_ids_2, b_type_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                       batch['token_type_ids_1'], batch['attention_mask_1'], batch['token_ids_2'],
                                       batch['token_type_ids_2'], batch['attention_mask_2'], batch['labels'])
            b_ids_1 = b_ids_1.to(device)
            b_type_ids_1 = b_type_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_type_ids_2 = b_type_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            loss = F.cross_entropy(logits.float().squeeze(), b_labels.float().view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
        train_loss = train_loss / (num_batches)

        #French similarity
        for batch in tqdm(fr_similarity_train_dataloader, desc=f'trainSimFr-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_type_ids_1, b_mask_1, b_ids_2, b_type_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                       batch['token_type_ids_1'], batch['attention_mask_1'], batch['token_ids_2'],
                                       batch['token_type_ids_2'], batch['attention_mask_2'], batch['labels'])
            b_ids_1 = b_ids_1.to(device)
            b_type_ids_1 = b_type_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_type_ids_2 = b_type_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_similarity_fr(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            loss = F.cross_entropy(logits.float().squeeze(), b_labels.float().view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
        train_loss = train_loss / (num_batches)
        #BEFORE: Just evaluated on SST
        # train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        #NOW: Evaluates on all 3
        train_acc, train_f1, *_ = model_eval_multitask(sst_train_dataloader, fr_sentiment_train_dataloader,
                         para_train_dataloader, fr_paraphrase_train_dataloader,
                         sts_train_dataloader, fr_similarity_train_dataloader,
                         model, device)
        dev_acc, dev_f1, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)