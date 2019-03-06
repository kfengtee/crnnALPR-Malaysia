import collections
import itertools
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np

class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot

def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)

def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))

def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def max_idx(array):
    """
    Return the position of the largest value in an 2D-array.
    """
    indices = np.where(array==array.max())
    return list(zip(indices[0], indices[1]))

def seq_rank(mat, ranks):
    """
    Find the position of n-most probable sequences from a given matrix. 
    
    mat: the matrix containing probabilities 
    ranks: the "n" most probable sequence you need
    
    <KNOWN ISSUE>: mat must be float type (integer type will not work due to the usage of np.NINF)
    <KNOWN ISSUE>: the rank in "Only {rank} ranking is available" is calculated wrongly
    """
    seq = [] # empty list to store seq
    mat = mat.copy() # make a copy of given matrix (we don't want to change the ori mat)
    
    # number of ranking you need
    for rank in range(ranks):
        if np.all(mat==0):
            print(f"Only {rank} ranking is available")
            break
        
        # finding the most probable seq
        if rank == 0:
            first_seq = list(mat.argmax(0))
            seq.append(first_seq)

            # reset highest value to zero
            for i in enumerate(first_seq):
                mat[i[1], i[0]] = np.NINF

        # finding second most probable seq onwards
        if rank >= 1:
            # find the position of currently highest element
            idx_list = max_idx(mat)
            for idx in idx_list:
                # must get an copy of the last element of seq because new_seq = seq[-1] 
                # making a reference not a copy
                new_seq = seq[-1].copy()
                # e.g: idx = (2,3) suggest that seq[3] need to be replaced by 2
                new_seq[idx[1]] = idx[0]
                seq.append(new_seq)

                # reset the highest value to zero
                mat[idx] = np.NINF
    return np.asarray(seq)

def max_idx(mat):
    """
    Return the coordinate of the largest value in an 2D-array.
    
    Returns
    -------
    out : a list of tuples (coordinate)
    """
    indices = np.where(mat==mat.max())
    return list(zip(indices[0], indices[1]))

def dup_argmax_idx(mat):
    
    """
    Returns the indices of columns of duplicated prediction pairs (after removing "empty" prediction)
    NOTE: duplicated "empty" prediction pairs are ignored
    
    Returns
    -------
    out : a list containing all indices of duplicated prediction pairs columns 
    
    Notes
    -----
    
    """
    dup_col_idx = []

    first_occurance_idx = 0
    first_occurance_argmax = 0 

    for col_idx, argmax in enumerate(mat.argmax(0)):
        
        # we only consider if argmax is not 0 ("empty")
        if argmax != 0:
            # if the current argmax is same as immediate last argmax 
            if (argmax == first_occurance_argmax):
                # record down these duplicated columns index
                dup_col_idx = dup_col_idx + [first_occurance_idx] + [col_idx]

                # reset the first occurance index and value (this will ensure next loop goes to "else", 
                # i.e: setting next "non-empty" column as "first_occurance" 
                
                first_occurance_idx = 0
                first_occurance_argmax = 0 

            # current non-zero argmax is different from immediate last argmax
            else:
                # set this as new first occurance index and value
                first_occurance_idx = col_idx
                first_occurance_argmax = argmax

    return(dup_col_idx)


def pred_conf(out_mat, num_top_results):
    """
    Compute the top predicted sequences and their corresponding prediction confidence from the output matrix
    of neural network.
    
    Input
    -----
    out_mat: output matrix from crnn
    num_top_results: number of top prediction results
    
    Notes
    -----
    Assumption:
    1) Initial "empty" predictions are accurate
    2) Duplicated predictions (e.g "--W--W" > "W") are accurate 
    
    Returns
    -------
    out: a list of tuples containing (predicted sequence, prediction's confidence)
    
    """
    np.seterr(divide='ignore')
    
    out_mat = out_mat.copy() # to avoid changing the original output matrix
    seq = [] # predicted sequence (before decoding to alphanumeric)
    seq_act = [] # corresponding activations of the predicted sequence
    logits_sum = np.exp(out_mat).sum(axis=0, dtype='f') # sum(exp(score))
    
    
    
    # top predicted sequence
    top_seq = list(out_mat.argmax(0))
    seq.append(top_seq)
    
    
    
    # PRELIMINARY FILTERING (based on the assumptions)
    # Criteria 1: ignore "empty" prediction
    # Criterai 2: ignore "duplicate prediction"
    empty_col_idx = np.where(out_mat.argmax(0)==0)[0].tolist()
    dup_col_idx = dup_argmax_idx(out_mat)
    
    # set these columns to -INF
    out_mat[:, empty_col_idx + dup_col_idx] = np.NINF
    
    
    
    top_seq_act = []
    # Captures the activations of top predicted sequence and set it to -INF afterwards
    for col, argmax in enumerate(top_seq):
        if np.isfinite(out_mat[argmax, col]):
            top_seq_act.append(out_mat[argmax, col])
            out_mat[argmax, col] = np.NINF
        else:
            top_seq_act.append(np.nan)
    
    seq_act.append(top_seq_act)
    
    
    
    # For second top predictions onward
    for top_n_result in range(1, num_top_results):
        # if the entire out_mat become -inf, there isn't more possible predictions
        if np.all(~np.isfinite(out_mat)):
            print(f"Only {top_n_result} possible results")
            break
        
        # find the coordinates of the next most probable predictions in output matrix (after "PRELIMINARY FILTERING")
        # we're interested to know the position of the current "highest" activation in the output matrix
        coord = max_idx(out_mat)
        for (argmax, col) in coord:
            
            # the next most probable sequence is the same as immediate last sequence, EXCEPT for 1 character, 
            # which is the character in the position of max_idx
            next_seq = seq[-1].copy()
            next_seq_act = seq_act[-1].copy()
            
            # change whatever character in the position to character indicated by max_idx
            next_seq[col] = argmax
            next_seq_act[col] = out_mat[argmax, col]
            
            seq.append(next_seq)
            seq_act.append(next_seq_act)

            # reset the current highest activation to -NINF 
            out_mat[(argmax, col)] = np.NINF
            
            
            
    # calculate logits from the activations (logits = exp(act))
    logits = np.exp(seq_act)
    # individual prediction probability (prob = logit / sum(logits) in the same column) (softmax)
    prob_ind = np.divide(logits, logits_sum)
    
    ## EXTREME CASES CONSIDERATION
    
    # 1) prob_ind contains infinity values
    # - happens when sum(logits) = 0
    prob_ind[np.isinf(prob_ind)] = np.nan
    
    # 2) entire prob_ind matrix is np.nan
    # - happens when input image is irrelevant 
    # - many "empty" => np.nan && "non-empty columns" has sum(logits) == 0 (Extreme case 1)
    if np.all(np.isnan(prob_ind)):
        prob_total = np.zeros(num_top_results)
    else:
        prob_total = np.nanprod(prob_ind, axis=1)
        
        # 3) final probability is more than 1 
        # - rounding error could happen
        prob_total[prob_total>1] = 1
    
    
    return list(zip(np.asarray(seq), prob_total))