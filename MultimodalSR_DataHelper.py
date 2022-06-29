'''
helper classes to process train, validation and test datasets for SASrecsys model
author: bazman, 2021
'''
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import random

from tqdm.auto import tqdm 

class SequenceDataValidationFullLength(torch.utils.data.Dataset):
    '''
    Dataset for validation - similar to SequenceDataValidation calss but puts all items except validation into validation sequence
    So it can be used for true NDCG HIT rate metrics rather than sampling 100 items in validation sequence
    train -> **valid** -> test
    dataset to produce validation data
    Input:
    - user_train : known sequence of items for the user (train data)
    - user_valid : one item that makes up a next selection after user_train sequence
    - usernum : number of users in user_train/user_valid
    - itemnum : number of items in user_train/user_valid
    - maxlen : max length of sequence
    Returns:
    - user_train is the same
    - user_valid is appended with all items except for vlidation item and after that all those items are scored with model and logit 
        for the 0-th element(user_valid) should be somewhere in top 10 or 100 scores
    '''
    def __init__(self, user_train, user_valid, usernum, itemnum, maxlen):
        '''
        Input:
        - user_train: dict of user training sequence
        - user_valid: dict with one item for validation sequence
        - usernum - number of users in dataset
        - itemnum - number of items in dataset
        - maxlen - max len of sequence for truncation
        - ndcg_samples - how many random items do we sample to calculate hit rate and ndcg
        Output:
        self.seq - maxlen sequnce for train
        self.valid - 101 len for validation
        '''
        from tqdm import tqdm
        super(SequenceDataValidationFullLength, self).__init__()
        
        # make a list of users to validate on
        # limit users max to 10000 or to whatever we have in case less than 10000
        if usernum > 10_000:
            users = random.sample(range(1, usernum + 1), 10_000)
        else:
            users = range(1, usernum + 1)
            
        # making a validation sequence with one element from valid and the rest random
        # all elements that are in train plus padding zero
        valid_seq = torch.zeros((len(users), itemnum), dtype=torch.int)
        
        # make a matrix from train sequence (batch, maxlen)
        final_seq = torch.zeros((len(users), maxlen), dtype=torch.int)
        
        with tqdm(total=len(users)) as pbar:
            for ii,_u in enumerate(users):
                # truncate seq  to maxlen
                idx = min(maxlen, len(user_train[_u]))
                final_seq[ii, -idx:] = torch.as_tensor(user_train[_u][-idx:])
                
                all_items_set = set(range(1, itemnum+1)) # set of all possible items
                validation_items_set = all_items_set - set(user_valid[_u])
            
                valid_seq[ii,0] = user_valid[_u][0] # get true next element from validation set
                valid_seq[ii,1:] = torch.from_numpy(np.array(list(validation_items_set))) # all items except validation one
                pbar.update(1)
        
        self.seq = final_seq # store training seq
        self.valid = valid_seq # store validation seq
        self.users = users # store validation users
            
    def __getitem__(self, index):
        return self.seq[index], self.valid[index]

    def __len__(self):
        return len(self.seq)


class SequenceDataValidation(torch.utils.data.Dataset):
    '''
    Dataset for validation
    train -> **valid** -> test
    dataset to produce validation data
    Input:
    - user_train : known sequence of items for the user (train data)
    - user_valid : one item that makes up a next selection after user_train sequence
    Returns:
    - user_train is the same
    - user_valid is appended with 100 random items that are not in user_trian after that 101 items are scored with model and logit 
        for the 0-th element(user_valid) should be somewhere in top 10 scores
    '''
    def __init__(self, user_train, user_valid, usernum, itemnum, maxlen, pre_image_name, pre_text_name, ndcg_samples=100):
        '''
        Input:
        - user_train: dict of user training sequence
        - user_valid: dict with one item for validation sequence
        - usernum - number of users in dataset
        - itemnum - number of items in dataset
        - maxlen - max len of sequence for truncation
        - ndcg_samples - how many random items do we sample to calculate hit rate and ndcg
        Output:
        self.seq - maxlen sequnce for train
        self.valid - 101 len for validation
        '''
        from tqdm import tqdm
        super(SequenceDataValidation, self).__init__()
        
        # make a list of users to validate on
        # limit users max to 10000 or to whatever we have in case less than 10000
        if usernum > 10_000:
            users = random.sample(range(1, usernum + 1), 10_000)
        else:
            users = range(1, usernum + 1)
            
        # making a validation sequence with one element from valid and the rest random
        # all elements that are in train plus padding zero
        valid_seq = torch.zeros((len(users), ndcg_samples+1), dtype=torch.int)
        
        # make a matrix from train sequence (batch, maxlen)
        final_seq = torch.zeros((len(users), maxlen), dtype=torch.int)
        
        with tqdm(total=len(users)) as pbar:
            for ii,_u in enumerate(users):
                # truncate seq  to maxlen
                idx = min(maxlen, len(user_train[_u]))
                final_seq[ii, -idx:] = torch.as_tensor(user_train[_u][-idx:])

                items_not_in_seq = np.array(list(set(range(1,itemnum+1)) - set(final_seq[ii].numpy().flatten()))) # random stuff not in final_seq
                valid_seq[ii,0] = user_valid[_u][0] # get true next element from validation set
                valid_seq[ii,1:] = torch.from_numpy(items_not_in_seq[np.random.randint(0, len(items_not_in_seq), ndcg_samples)]) # fill the rest with random stuff
                pbar.update(1)
        
        self.seq = final_seq # store training seq
        self.valid = valid_seq # store validation seq
        self.users = users # store validation users
        
        self.image_feature = np.load(f'../data/Amazon_2018/{pre_image_name}.npy')
        self.image_feature = np.concatenate((np.zeros((1, 4096)), self.image_feature), axis = 0) # 0번 아이템 제로패딩
        
        self.text_feature = np.load(f'../data/Amazon_2018/{pre_text_name}.npy')
        self.text_feature = np.concatenate((np.zeros((1, 768)), self.text_feature), axis = 0)
            
    def __getitem__(self, index):
        return self.seq[index], self.valid[index], self.image_feature[self.seq[index]], self.text_feature[self.seq[index]]

    def __len__(self):
        return len(self.seq)

class SequenceDataTest(SequenceDataValidation):
    '''
    Dataset for test
    train -> valid -> **test**
    dataset to produce test data set
    same as SequenceDataValidation class but uses one element from test_seq to make a test_seq
    alse adds up validation item to train sequence
    '''
    def __init__(self, user_train, user_valid, user_test, usernum, itemnum, maxlen, pre_image_name, pre_text_name, ndcg_samples):
        super().__init__(user_train, user_test, usernum, itemnum, maxlen,pre_image_name, pre_text_name, ndcg_samples)
        # now we need to shift self.seq one item back
        self.seq[:,:-1] = self.seq[:,1:]
        # this is an extra item that will be the last in training seq
        extra_valid_item = torch.as_tensor([user_valid[_u][0] for _u in self.users])
        self.seq[:,-1] = extra_valid_item
        

class SequenceData(torch.utils.data.Dataset):
    '''
    dataset for training the network
    '''
    def __init__(self, user_seq, usernum, itemnum, pre_image_name, pre_text_name):
        '''
        user_seq is a dict with keys = userid - sequential from 1 to number of users(usernum)
        itemnum - number of items in vocabulary of selected movies
        Sets up the following props in the object:
        seq - all elements of user seq without last element
        pos - all elements without first element (shift one time item ahead)
        neg - the same length but with all different elements
        Resulting data looks like this:
        seq = [250,  13, 251,  70, 252,  81, 237, 150, 253,  27, 143, 254, 236,
        196, 229, 255, 256, 179, 167, 172, 157, 257,  39, 199, 258]
        
        pos = [ 13, 251,  70, 252,  81, 237, 150, 253,  27, 143, 254, 236, 196,
        229, 255, 256, 179, 167, 172, 157, 257,  39, 199, 258,  29]
        
        neg =  [928, 3404,  821, 2505, 1931, 2588, 1365,  527, 3140, 1615, 1649,
        1981,  450, 1175, 1576, 1787, 1425, 2698, 1916,  729, 3390, 2503,
        2751, 1481, 2422]
        '''
        from tqdm import tqdm # 맨위로 보내기
        super(SequenceData, self).__init__()
        self.usernum = usernum
        self.userids = np.array(list(user_seq.keys())) # store userids in a property
        self.seq, self.pos, self.neg = dict(), dict(), dict()
        with tqdm(total=len(user_seq)) as pbar:
            for userid, _user_seq in user_seq.items():
                self.seq[userid] = np.array(_user_seq[:-1]) # all but last element
                self.pos[userid] = np.array(_user_seq[1:]) # shifted one time slot ahead
                # negative sequence
                items_not_in_seq = np.array(list(set(range(1,itemnum+1)) - set(_user_seq))) # all items from vocab that are out of user_seq
                self.neg[userid] = items_not_in_seq[np.random.randint(0, len(items_not_in_seq), len(self.seq[userid]))] # select random items from above array
                pbar.update(1)
                
        self.image_feature = np.load(f'../data/Amazon_2018/{pre_image_name}.npy')
        self.image_feature = np.concatenate((np.zeros((1, 4096)), self.image_feature), axis = 0) # 0번 아이템 제로패딩
        
        self.text_feature = np.load(f'../data/Amazon_2018/{pre_text_name}.npy')
        self.text_feature = np.concatenate((np.zeros((1, 768)), self.text_feature), axis = 0)
    def __getitem__(self, index):
        userid = self.userids[index]
        return [userid, self.seq[userid], self.pos[userid], self.neg[userid], self.image_feature[self.seq[userid]], self.text_feature[self.seq[userid]]]

    def __len__(self):
        return len(self.seq)

def tokenize_batch(batch, max_len=50):
    '''
    use tokenizer to cast dict type to tensors and shrink the data to maxlen - nothing else
    could have made it in dataset directly but anyway...
    '''
    u        = []
    seq_list = []
    pos_list = []
    neg_list = []
    
    image_list = []
    text_list = []
    
    for _u, seq, pos, neg, image_feature, text_feature in batch:
        # fixed size tensor of max_len
        seq_holder = torch.zeros(max_len, dtype=torch.int)
        pos_holder  = torch.zeros_like(seq_holder)
        neg_holder = torch.zeros_like(seq_holder)
        
        idx = min(max_len, len(seq))
        seq_holder[-idx:] = torch.from_numpy(seq[-idx:])
        pos_holder[-idx:] = torch.from_numpy(pos[-idx:])
        neg_holder[-idx:] = torch.from_numpy(neg[-idx:])
        
        seq_list.append(seq_holder.unsqueeze(dim=0))
        pos_list.append(pos_holder.unsqueeze(dim=0))
        neg_list.append(neg_holder.unsqueeze(dim=0))
        u.append(_u)
        
        # side infomation
        image_feature = np.concatenate((np.zeros(((max_len - idx), 4096)), image_feature), axis = 0)[-50:]
        text_feature = np.concatenate((np.zeros(((max_len - idx), 768)), text_feature), axis = 0)[-50:]
        
        image_list.append(torch.from_numpy(image_feature).unsqueeze(dim=0))
        text_list.append(torch.from_numpy(text_feature).unsqueeze(dim=0))
       
    return u, torch.cat(seq_list, dim=0), torch.cat(pos_list, dim=0), torch.cat(neg_list, dim=0), torch.cat(image_list, dim=0), torch.cat(text_list, dim=0)


# train/val/test data generation
def data_partition(fname):
    '''
    Partition the data into train, valid and test sets. 
    Input :  file in format 
    user_id<space>item_selected
    ...
    user_id<space>item_selected
    All items appear according to time order
    Returns:
    user_train - dict with key = userid and value = list of all items selected in respected time order
    user_valid - dict with the same structure as above but with penulitimate item (just one item)
    user_test - same as above but with ultimate item selected
    i.e. you have user 5 with items 1,29,34,15,8 there will be 
    user_train[5] = [1,29,34], user_valid = [15], user_test=[8]
    usernum - number of users
    itemnum - number of items
    '''
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    f = open('data/%s.txt' % fname, 'r')

    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
                
    return [user_train, user_valid, user_test, usernum, itemnum]