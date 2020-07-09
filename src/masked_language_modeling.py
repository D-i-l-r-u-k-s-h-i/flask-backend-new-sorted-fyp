from src import bert_config
import torch
import re
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
import language_tool_python
import spacy
import textacy
from spacy.gold import biluo_tags_from_offsets

nlp = spacy.load('en')


def fill_masks(textData):
    text = textData
    text = '[CLS] ' + text + ' [SEP]'
    print("text fill_mask: " + text)
    tokenized_text = bert_config.BERT_TOKENIZER.tokenize(text)
    print("tokenized_text: " + str(tokenized_text))
    indexed_tokens = bert_config.BERT_TOKENIZER.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        predictions = bert_config.BERT_MODEL(tokens_tensor, segments_tensors)[0]

    results = []
    for i, t in enumerate(tokenized_text):

        if t == '[MASK]':
            # Get 10 choices for the masked(blank) word
            k = 10
            predicted_index, predicted_index_values = torch.topk(predictions[0, i], k)
            predicted_tokens = bert_config.BERT_TOKENIZER.convert_ids_to_tokens(predicted_index_values.tolist())
            print(predicted_tokens)

            filtered_tokens_to_remove_punctuation = []
            # get only the predictions that doesn't contain punctuation.
            for token in predicted_tokens:
                if re.match("^[a-zA-Z0-9_]*$", token):
                    filtered_tokens_to_remove_punctuation.append(token)
            if len(filtered_tokens_to_remove_punctuation) is not 0:
                results.append(filtered_tokens_to_remove_punctuation[0])
            else:
                results.append('')

            print(filtered_tokens_to_remove_punctuation)
            print(results)
        results_iterator = iter(results)

    sentenceList = text
    for i, word in enumerate(text.split()):
        if '[MASK]' in word:
            sentenceList = sentenceList.replace('[MASK]', next(results_iterator), 1)

    comp_text = sentenceList.replace('[CLS] ', '').replace(' [SEP]', '')
    print(comp_text)
    return comp_text


def fill_predicted_masks(text):
    text = '[CLS] ' + text + ' [SEP]'
    tokenized_text = bert_config.BERT_TOKENIZER.tokenize(text)
    indexed_tokens = bert_config.BERT_TOKENIZER.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        predictions = bert_config.BERT_MODEL(tokens_tensor, segments_tensors)[0]

    for i, t in enumerate(tokenized_text):
        if t == '[MASK]':
            # Get 10 choices for the masked(blank) word
            k = 20
            predicted_index, predicted_index_values = torch.topk(predictions[0, i], k)
            predicted_tokens = bert_config.BERT_TOKENIZER.convert_ids_to_tokens(predicted_index_values.tolist())
            # print(predicted_tokens)

    return predicted_tokens


def completed_outcome_maskedLM(para):
    tool = language_tool_python.LanguageTool('en-US')

    sentences = sent_tokenize(para)
    sentence_list = []

    for sentence in sentences:
        print(sentence)

        spell_checked_sentence = tool.correct(sentence)

        #   check svos and svs
        grammar_checked_sent = grammer_check_and_add_suggestions(spell_checked_sentence)
        print("grammar_checked_sent: " + grammar_checked_sent)

        tokenized_sentence = word_tokenize(grammar_checked_sent)

        tokens_tag = pos_tag(tokenized_sentence)

        list_of_dict = []

        pos_list2 = ["JJ", "JJR", "JJS", "MD", "VB", "VBG", "VBD", "WP", "WRB", "WDT", "VBN", "VBP", "PRP", "PRP$",
                     "RBR", "RBS", "RP", "TO"]

        BILUO_list = getBILUO(grammar_checked_sent)

        for idx, tupl in enumerate(tokens_tag):
            # print(tokens_tag[idx][1])
            pos_list = ["FW", "JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR",
                        "RBS", "CD", "DT", "IN", "."]
            if tokens_tag[idx][1] not in pos_list:
                replacement = '[MASK]'

                temp_sentence = grammar_checked_sent
                temp_sentence = word_tokenize(temp_sentence)

                temp_sentence[idx] = replacement

                predicted_tokens = fill_predicted_masks(' '.join(temp_sentence))

                if tupl[0] not in predicted_tokens:
                    dict = {'word': tupl[0], 'index': idx}
                    list_of_dict.append(dict)

            elif BILUO_list[idx].startswith('L') or BILUO_list[idx].startswith('U'):
                print(BILUO_list)
                print("BILUO_list outer")
                if (len(BILUO_list) > idx + 1) and (
                        BILUO_list[idx + 1].startswith('B') or BILUO_list[idx + 1].startswith('U')):
                    print("BILUO_list inner")
                    dict = {'word': tupl[0], 'index': idx}
                    list_of_dict.append(dict)
        # print(list_of_dict)

        for word in enumerate(list_of_dict):
            # print(word[1])
            if tokens_tag[word[1]['index']][1] in pos_list2:
                print(word[1]['word'])
                tokenized_sentence[word[1]['index']] = word[1]['word'] + " [MASK]"
            else:
                print(word[1]['word'])
                tokenized_sentence[word[1]['index']] = word[1]['word'] + " [MASK]" + " [MASK]"

        joined_sentence = ' '.join(tokenized_sentence)
        if "[MASK]" in joined_sentence:
            completed_text = fill_masks(joined_sentence)
            sentence_list.append(completed_text)
        else:
            sentence_list.append(joined_sentence)

    print(' '.join(sentence_list))
    return ' '.join(sentence_list)


# sentence pattern check----------------------------------------------------------------------------------------------

SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]


def findSVO(sentence):
    text = nlp(sentence)

    text_ext = textacy.extract.subject_verb_object_triples(text)

    return list(text_ext)


def getAllSubs(v):
    verbNegated = isNegated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
    return subs, verbNegated


def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs


def findSubs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == "NOUN":
        return [head], isNegated(tok)
    return [], False


def isNegated(tok):
    negations = {"no", "not", "n't", "ever", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False


def findSV(text):
    tokens = nlp(text)
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]

    for v in verbs:
        subs, verbNegated = getAllSubs(v)

        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.lower_, "!" + v.lemma_ if verbNegated else v.lemma_))
    return svs


def getNERDiff(doc):
    # doc = nlp(token)
    token_label = []

    for i, ent in enumerate(doc.ents):
        # check for entity in string
        # print(ent.text, ent.start_char, ent.end_char,ent.label_+"-"*10)
        token_label.append((ent.start_char, ent.end_char, ent.label_))
    return token_label


def getBILUO(token):
    doc1 = nlp(token)
    entities = getNERDiff(doc1)
    tags = biluo_tags_from_offsets(doc1, entities)
    return tags


# get Named Entities

def getNER(token):
    doc = nlp(token)
    token_label = []

    for i, ent in enumerate(doc.ents):
        # check for entity in string
        # print(ent.text, ent.start_char, ent.end_char,ent.label_+"-"*10)
        token_label.append((ent.text, ent.label_))
    return token_label


# get word dependencies according to the sentence

def getDependancies(token):
    doc = nlp(token)
    dependancy_list = []

    # dependancy wrt sentence
    for token in doc:
        # print (token.text, token.tag_, token.head.text, token.dep_+"-"*5)
        dependancy_list.append((token.text, token.dep_))

    return dependancy_list


# Grammar suggestions before masking

def grammer_check_and_add_suggestions(text):
    # print(text)
    if len(findSVO(text)) is not 0:
        print(findSVO(text))
        print("findSVO is not None")
        sent = text
    elif len(findSV(text)) is not 0:
        print(findSV(text))
        print("len(findSV) is not 0")
        sent = text
    else:
        # when there is no grammar pattern
        print("else")

        list_of_dict = []
        list_result = []

        list_NER = getNER(text)
        list_Dep = getDependancies(text)

        for ner_item in list_NER:
            # print(ner_item[1])
            result = [i + 1 for i, word in enumerate(list_Dep) if word[0] in ner_item[0]]
            list_result.append(result)
            # print(result)
            for i, word in enumerate(list_Dep):
                if word[0] in ner_item[0] and word[1] is not "dobj":
                    # to send the index starting count from 1 not 0
                    dict = {'word': ner_item[0], 'index': i + 1}
                    list_of_dict.append(dict)

        for i, w_lst in enumerate(list_result):
            # check consecutive entities
            if (sorted(w_lst)) == list(range(min(w_lst), max(w_lst) + 1)):
                for i, word in enumerate(list_of_dict):
                    if word['index'] == max(w_lst):
                        text = text.replace(word['word'], word['word'] + " [MASK]", 1)
                        print("text: " + text)
                        break

        sent = fill_masks(text)
        print("sent: " + sent)
    return sent
