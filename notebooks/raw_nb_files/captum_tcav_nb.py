{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Testing with Concept Activation Vectors (TCAV) on Sensitivity Classification Examples and a ConvNet model trained on IMDB DataSet"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This tutorial shows how to apply TCAV, a concept-based model interpretability algorithm, on sentiment classification task using a ConvNet model (https://captum.ai/tutorials/IMDB_TorchText_Interpret) that was trained using IMDB sensitivity dataset.\n",
    "\n",
    "More details about the approach can be found here: https://arxiv.org/pdf/1711.11279.pdf\n",
    "\n",
    "In order to use TCAV, we need to predefine a list of concepts that we want our predictions to be test against.\n",
    "\n",
    "Concepts are human-understandable, high-level abstractions such as visually represented \"stripes\" in case of images or \"positive adjective concept\" such as \"amazing, great, etc\" in case of text. Concepts are formatted and represented as input tensors and do not need to be part of the training or test datasets.\n",
    "\n",
    "Concepts are incorporated into the importance score computations using Concept Activation Vectors (CAVs). Traditionally, CAVs train linear classifiers and learn decision boundaries between different concepts using the activations of predefined concepts in a NN layer as inputs to the classifier that we train. The vector that is orthogonal to learnt decision boundary and is pointing towards the direction of a concept is the CAV of that concept.\n",
    "\n",
    "TCAV measures the importance of a concept for a prediction based on the directional sensitivity (derivatives) of a concept in Neural Network (NN) layers. For a given concept and layer it is obtained by aggregating the dot product between CAV for given concept in given layer and the gradients of model predictions w.r.t. given layer output. The aggregation can be performed based on either signs or magnitudes of the directional sensitivities of concepts across multiple examples belonging to a certain class. More details about the technique can be found in above mentioned papers.\n",
    "\n",
    "Note: Before running this tutorial, please install the spacy, numpy, scipy, sklearn, PIL, and matplotlib packages.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import glob\n",
    "import random\n",
    "\n",
    "import torch\n",
    "import torchtext\n",
    "import spacy\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import numpy as np\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from scipy.stats import ttest_ind\n",
    "\n",
    "from torch.utils.data import DataLoader, Dataset, IterableDataset\n",
    "\n",
    "#.... Captum imports..................\n",
    "from captum.concept import TCAV\n",
    "from captum.concept import Concept\n",
    "from captum.concept._utils.common import concepts_to_str\n",
    "\n",
    "from torchtext.vocab import Vocab\n",
    "\n",
    "from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset\n",
    "\n",
    "nlp = spacy.load('en')\n",
    "\n",
    "# fixing the seed for CAV training purposes and performing train/test split\n",
    "random.seed(123)\n",
    "np.random.seed(123)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": []
    }
   ],
   "source": [
    "device = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Defining torchtext data `Field` so that we can load the vocabulary for IMDB dataset the way that was done to train IMDB model. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": []
    }
   ],
   "source": [
    "TEXT = torchtext.data.Field(lower=True, tokenize='spacy')\n",
    "Label = torchtext.data.LabelField(dtype = torch.float)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Reading IMDB dataset the same way we did for training sensitivity analysis model. This will help us to load correct token to embedding mapping using Glove."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": []
    }
   ],
   "source": [
    "train, _ = torchtext.datasets.IMDB.splits(text_field=TEXT,\n",
    "                                              label_field=Label,\n",
    "                                              train='train',\n",
    "                                              test='test',\n",
    "                                              path='data/aclImdb')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Loading token-to-embedding vectors from Glove and building the vocabulary using IMDB training dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "from torchtext import vocab\n",
    "\n",
    "MAX_VOCAB_SIZE = 25_000\n",
    "\n",
    "# If you prefer to use pre-downloaded glove vectors, you can load them with the following two command line\n",
    "loaded_vectors = torchtext.vocab.Vectors('data/glove.6B.50d.txt')\n",
    "\n",
    "TEXT.build_vocab(train, vectors=loaded_vectors, max_size=len(loaded_vectors.stoi))\n",
    "TEXT.vocab.set_vectors(stoi=loaded_vectors.stoi, vectors=loaded_vectors.vectors, dim=loaded_vectors.dim)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Vocabulary Size:  101513\n"
     ]
    }
   ],
   "source": [
    "print('Vocabulary Size: ', len(TEXT.vocab))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A helper function that allows to read concept examples from an input file. We also define `const_len=7`, minimum and maximum length of tokens in text. The text is extended up to 7 tokens with padding, if it is short or is truncated up to 7 tokens if it is long."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_tensor_from_filename(filename):\n",
    "    ds = torchtext.data.TabularDataset(path=filename,\n",
    "                                       fields=[('text', torchtext.data.Field()),\n",
    "                                               ('label', torchtext.data.Field())],\n",
    "                                       format='csv')\n",
    "    const_len = 7\n",
    "    for concept in ds:\n",
    "        concept.text = concept.text[:const_len]\n",
    "        concept.text += ['pad'] * max(0, const_len - len(concept.text))\n",
    "        text_indices = torch.tensor([TEXT.vocab.stoi[t] for t in concept.text], device=device)\n",
    "        yield text_indices\n",
    "        "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The function below allows us to create a concept instance using a file path where the concepts are stored, concept name and id."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def assemble_concept(name, id, concepts_path=\"data/tcav/text-sensitivity\"):\n",
    "    dataset = CustomIterableDataset(get_tensor_from_filename, concepts_path)\n",
    "    concept_iter = dataset_to_dataloader(dataset, batch_size=1)\n",
    "    return Concept(id=id, name=name, data_iter=concept_iter)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's define and visualize the concepts that we'd like to explore in this tutorial.\n",
    "\n",
    "For this tutorial we look into two concepts, `Positive Adjectives` and `Neutral`. `Positive Adjectives` concept defines a group of adjectives that convey positive emotions such as `good` or `lovely`. The `Neutral` concept spans broader domains / subjects and is distinct from the `Positive Adjectives` concept.\n",
    "\n",
    "The concept definition and the examples describing that concepts are left up to the user.\n",
    "\n",
    "Below we visualize examples from both `Positive Adjectives` and `Neutral` concepts. This concepts are curated for demonstration purposes. It's up to a user what to include into a concept and what not."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_concept_sample(concept_iter):\n",
    "    cnt = 0\n",
    "    max_print = 10\n",
    "    item = next(concept_iter)\n",
    "    while cnt < max_print and item is not None:\n",
    "        print(' '.join([TEXT.vocab.itos[item_elem] for item_elem in item[0]]))\n",
    "        item = next(concept_iter)\n",
    "        cnt += 1\n",
    "\n",
    "neutral_concept = assemble_concept('neutral', 0, concepts_path=\"data/tcav/text-sensitivity/neutral.csv\")\n",
    "positive_concept = assemble_concept('positive-adjectives', 5, \\\n",
    "                                    concepts_path=\"data/tcav/text-sensitivity/positive-adjectives.csv\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Both `Positive Adjective` and `Neutral` concepts have the same number of examples representing corresponding concept. The examples for the `Positive Adjective` concept are semi-hand curated and the context is neutralized whereas those for neutral concept are chosen randomly from Gutenberg Poem Dataset (https://github.com/google-research-datasets/poem-sentiment/blob/master/data/train.tsv)\n",
    "\n",
    "You can consider also using Stanford Sentiment Tree Bank (SST, https://nlp.stanford.edu/sentiment/index.html) dataset with `neutral` labels. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Examples representing `Neutral` Concept"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "with pale blue <unk> in these peaceful\n",
      "it flows so long as falls the\n",
      "and that is why pad pad pad\n",
      "when i peruse the conquered fame of\n",
      "of inward strife for truth and <unk>\n",
      "the red sword sealed their <unk> pad\n",
      "and very venus of a <unk> pad\n",
      "who the man pad pad pad pad\n",
      "and so <unk> then a worthless <unk>\n",
      "to hide the orb of <unk> every\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": []
    }
   ],
   "source": [
    "print_concept_sample(iter(neutral_concept.data_iter))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Examples representing `Positive Adjective` Concept"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "`Positive Adjectives` concept is surrounded by neutral / uninformative context. It is important to note that we positioned positive adjectives in different locations in the text. This makes concept definitions more abstract and independent of the location. Apart from that as we can see the length of the text in the concepts in fixed to 7. This is because our sensitivity classifier was trained for a fixed sequence length of 7. Besides that this ensures that the activations for concept and prediction examples have the same shape."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      ". . so well . . .\n",
      ". . so good . . .\n",
      ". . . love it . pad\n",
      ". . . like it . pad\n",
      ". . even greater . . .\n",
      ". antic . . . . .\n",
      ". . . fantastical . . .\n",
      ". grotesque . . . . .\n",
      "fantastic . . . . . pad\n",
      "grand . . . . . pad\n"
     ]
    }
   ],
   "source": [
    "print_concept_sample(iter(positive_concept.data_iter))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Defining and loading pre-trained ConvNet model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Defining the model, so that we can load associated weights into the memory."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "class CNN(nn.Module):\n",
    "    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, \n",
    "                 dropout, pad_idx):\n",
    "        \n",
    "        super().__init__()\n",
    "                \n",
    "        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)\n",
    "        \n",
    "        self.convs = nn.ModuleList([\n",
    "                                    nn.Conv2d(in_channels = 1, \n",
    "                                              out_channels = n_filters, \n",
    "                                              kernel_size = (fs, embedding_dim)) \n",
    "                                    for fs in filter_sizes\n",
    "                                    ])\n",
    "        \n",
    "        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)\n",
    "\n",
    "        self.dropout = nn.Dropout(dropout)\n",
    "        \n",
    "    def forward(self, text):\n",
    "        \n",
    "        #text = [sent len, batch size]\n",
    "        \n",
    "        #text = text.permute(1, 0)\n",
    "                \n",
    "        #text = [batch size, sent len]\n",
    "        \n",
    "        embedded = self.embedding(text)\n",
    "\n",
    "        #embedded = [batch size, sent len, emb dim]\n",
    "        \n",
    "        embedded = embedded.unsqueeze(1)\n",
    "        \n",
    "        #embedded = [batch size, 1, sent len, emb dim]\n",
    "        \n",
    "        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]\n",
    "            \n",
    "        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]\n",
    "                \n",
    "        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]\n",
    "        \n",
    "        #pooled_n = [batch size, n_filters]\n",
    "        \n",
    "        cat = self.dropout(torch.cat(pooled, dim = 1))\n",
    "\n",
    "        #cat = [batch size, n_filters * len(filter_sizes)]\n",
    "            \n",
    "        return self.fc(cat)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": []
    },
    {
     "data": {
      "text/plain": [
       "CNN(\n",
       "  (embedding): Embedding(101982, 50, padding_idx=1)\n",
       "  (convs): ModuleList(\n",
       "    (0): Conv2d(1, 100, kernel_size=(3, 50), stride=(1, 1))\n",
       "    (1): Conv2d(1, 100, kernel_size=(4, 50), stride=(1, 1))\n",
       "    (2): Conv2d(1, 100, kernel_size=(5, 50), stride=(1, 1))\n",
       "  )\n",
       "  (fc): Linear(in_features=300, out_features=1, bias=True)\n",
       "  (relu): ReLU()\n",
       "  (dropout): Dropout(p=0.5, inplace=False)\n",
       ")"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = torch.load('models/imdb-model-cnn-large.pt')\n",
    "model.eval()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Computing TCAV Scores"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Before computing TCAV scores let's created instances of `positive-adjectives` and `neutral` concepts.\n",
    "\n",
    "In order to estimate significant importance of a concept using two-sided hypothesis testing, we define a number of `neutral` concepts. All `neutral` concepts are defined using random samples from Gutenberg Poem Training Dataset (https://github.com/google-research-datasets/poem-sentiment/blob/master/data/train.tsv)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "neutral_concept = assemble_concept('neutral', 0, concepts_path=\"data/tcav/text-sensitivity/neutral.csv\")\n",
    "neutral_concept2 = assemble_concept('neutral2', 1, concepts_path=\"data/tcav/text-sensitivity/neutral2.csv\")\n",
    "neutral_concept3 = assemble_concept('neutral3', 2, concepts_path=\"data/tcav/text-sensitivity/neutral3.csv\")\n",
    "neutral_concept4 = assemble_concept('neutral4', 3, concepts_path=\"data/tcav/text-sensitivity/neutral4.csv\")\n",
    "neutral_concept5 = assemble_concept('neutral5', 4, concepts_path=\"data/tcav/text-sensitivity/neutral5.csv\")\n",
    "\n",
    "positive_concept = assemble_concept('positive-adjectives', 5, \\\n",
    "                                    concepts_path=\"data/tcav/text-sensitivity/positive-adjectives.csv\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Below we define five experimental sets consisting of `Positive Adjective` vs `Neutral` concept pairs. TCAV trains a model for each pair, and estimates tcav scores for each experimental set in given input layers. In this case we chose `convs.2` and `convs.1` layers. TCAV score indicates the importance of a concept in a given layer. The higher the TCAV score, the more important is that concept for given layer in making a prediction for a given set of samples."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": []
    }
   ],
   "source": [
    "experimental_sets=[[positive_concept, neutral_concept],\n",
    "                  [positive_concept, neutral_concept2],\n",
    "                  [positive_concept, neutral_concept3],\n",
    "                  [positive_concept, neutral_concept4],\n",
    "                  [positive_concept, neutral_concept5]]\n",
    "\n",
    "tcav = TCAV(model, layers=['convs.2', 'convs.1'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A helper function to convert text tokens into embedding indices. In other words numericalizing text tokens."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "def covert_text_to_tensor(input_texts):\n",
    "    input_tensors = []\n",
    "    for input_text in input_texts:\n",
    "        input_tensor = torch.tensor([TEXT.vocab.stoi[tok.text] for \\\n",
    "                                     tok in nlp.tokenizer(input_text)], device=device).unsqueeze(0)\n",
    "        input_tensors.append(input_tensor)\n",
    "    return torch.cat(input_tensors)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_scores(interpretations, layer_name, score_type, idx):\n",
    "    return [interpretations[key][layer_name][score_type][idx].item() for key in interpretations.keys()]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here we define a number of examples that contain positive sentiment and test the sensitivity of model predictions to `Positive Adjectives` concept."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "pos_input_texts = [\"It was a fantastic play ! pad\", \"A terrific film so far ! pad\", \"We loved that show ! pad pad\"]\n",
    "pos_input_text_indices = covert_text_to_tensor(pos_input_texts)\n",
    "\n",
    "\n",
    "positive_interpretations = tcav.interpret(pos_input_text_indices, experimental_sets=experimental_sets)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Auxiliary functions for visualizing TCAV scores."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "def format_float(f):\n",
    "    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))\n",
    "\n",
    "def plot_tcav_scores(experimental_sets, tcav_scores, layers = ['convs.2'], score_type='sign_count'):\n",
    "    fig, ax = plt.subplots(1, len(experimental_sets), figsize = (25, 7))\n",
    "\n",
    "    barWidth = 1 / (len(experimental_sets[0]) + 1)\n",
    "\n",
    "    for idx_es, concepts in enumerate(experimental_sets):\n",
    "        concepts = experimental_sets[idx_es]\n",
    "        concepts_key = concepts_to_str(concepts)\n",
    "        \n",
    "        layers = tcav_scores[concepts_key].keys()\n",
    "        pos = [np.arange(len(layers))]\n",
    "        for i in range(1, len(concepts)):\n",
    "            pos.append([(x + barWidth) for x in pos[i-1]])\n",
    "        _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)\n",
    "        for i in range(len(concepts)):\n",
    "            val = [format_float(scores[score_type][i]) for layer, scores in tcav_scores[concepts_key].items()]\n",
    "            _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)\n",
    "\n",
    "        # Add xticks on the middle of the group bars\n",
    "        _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)\n",
    "        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])\n",
    "        _ax.set_xticklabels(layers, fontsize=16)\n",
    "\n",
    "        # Create legend & Show graphic\n",
    "        _ax.legend(fontsize=16)\n",
    "\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the cell below we visualize TCAV scores for `Positive Adjective` and `Neutral` concepts in `convs.2` and `convs.1` layers. For this experiment we tested `Positive Adjective` concept vs 5 different `Neutral` concepts. As we can see, `Positive Adjective` concept has consistent high score across all layers and experimental sets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABZgAAAGzCAYAAACiiXIiAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAABGOElEQVR4nO3deZwcdZ0//tcnB0kQkkAIhwkx4RBFPMAgqAhBQS4FEQ+QQxAPEC/WXQF3v5IIv1XXFRDBW0RBBdYDFVgQlUvBXVC5BEEWUILIGUAgQBI+vz9mMk4uMlPppGd6ns/Hox+ZrqquenfNZ16pfnd1dam1BgAAAAAA+mtYuwsAAAAAAGBw0mAGAAAAAKARDWYAAAAAABrRYAYAAAAAoBENZgAAAAAAGhnRrg2vs846derUqe3aPLCCfvvb3z5Qa53Y7jqakkEweMkfoJ1kENBOMghol2fLn7Y1mKdOnZprrrmmXZsHVlAp5c/trmFFyCAYvOQP0E4yCGgnGQS0y7Plj0tkAAAAAADQiAYzAAAAAACNaDADAAAAANCIBjMAAAAAAI1oMAMAAAAA0IgGMwAAAAAAjYxodwEr26OPPpr77rsv8+bNa3cpMCiMHDky6667bsaOHdvuUga9J598Mvfff3+efPLJzJ8/v93lwIAnf1rLMRD0jwxqLRkEfTdixIiMHj06EydOzOjRo9tdzqDndRj0TyuOgTq6wfzoo4/m3nvvzaRJkzJmzJiUUtpdEgxotdbMnTs3d999d5J4gbUCHnnkkdx7772ZOHFi1l9//YwYMUIGwbOQP63lGAj6Rwa1lgyCvqu1Zv78+Xnsscfyl7/8Jeutt17GjRvX7rIGLa/DoH9adQzU0ZfIuO+++zJp0qSsvvrqAgX6oJSS1VdfPZMmTcp9993X7nIGtQceeCCTJ0/OWmutlZEjR8ogWA7501qOgaB/ZFBrySDou1JKRo4cmbXWWiuTJ0/Ogw8+2O6SBjWvw6B/WnUM1NEN5nnz5mXMmDHtLgMGnTFjxvg44wp6+umn5Q80IH9awzEQNCODWkMGQTNjxozJU0891e4yBjWvw6CZFT0G6ugGcxLvVkED/m5aw36E/vN30zr2JfSfv5vWsS+h//zdtIb9CP23on83Hd9gBgAAAABg5dBgBgAAAACgkeU2mEspp5VS7iul3LiM+aWUcnIp5bZSyvWllK1aX2ZrPTlvwZDePgwmMqjztg+DiQzqvO3DYCF/Om/7MJjIoM7bPqxMI/qwzOlJTkny7WXM3y3Jpt23bZJ8qfvfAWv0yOGZevT5bdv+nZ/eo23b7o9LL700O+64Yy655JLMmDEjSXLSSSdlypQpefOb37zIsjNnzsysWbNSa21Dpc0s7fkt/PfSSy9t+fZOP/30PPPMM3nXu961xPRDDjkkd9xxR6ZOndry7XaA0yODWmowZJD8aS35s0JOjwxqKRnUfjJo0Dg98qel5E/7yZ9B5fTIoJaSQe0ng1ae5TaYa62Xl1KmPssieyX5du0aUb8ppYwvpWxQa72nVUXSHltttVWuuuqqbL755j3TTjrppGy33XZLBMu73/3u7Lrrrqu6xJb74he/uNLWffrpp2f+/PlLBMsee+yRq666KhtssMFK2/ZgJoOGJvnTWvKnORk0NMmg1pJBzcifoUn+tJb8aU4GDU0yqLWGUgb15Qzm5ZmU5K5e92d3T1siVEop703y3iSZMmVKnzfw5LwFGT1y+IpVSb+NHTs22267bZ+WnTx5ciZPnrySK1r5eofoqjJx4sRMnDhxlW+3g8igDiR/Vg350xJ9yiD5M7gM1Ax65pmaYcNW7Bu+l6XVGdSXWmXQClvpx0CsevKnNZZXr/xpCa/DOpAMao2hmEGt+JK/pe2xpZ4fX2v9aq11eq11en925MKPMfT3NnvO3Fw/++ElboPZzJkzU0rJDTfckB133DGrr756Nthgg3ziE5/IM88807PcLbfckr333jvjx4/PmDFjsu222+bCCy9cZF233npr9t5776y77roZPXp0pkyZkre+9a2ZP39+kq6PB5RSej4mMHXq1Pz5z3/Od77znZRSUkrJwQcfvEhdC73oRS/KPvvss0T9//M//5NSSs4999yeadddd1323HPPrLXWWhkzZkxe/epX54orrujT/jjrrLPy2te+NhMnTswaa6yRLbfcMt/61reWWO7+++/PO97xjowdOzbjx4/PQQcdlIcffniJ5WbMmNHz8YiFHnjggRx++OGZNGlSRo0alRe84AX56le/usRj77jjjhx44IFZf/31M2rUqGy00Ub58Ic/3LPeyy67LL/+9a979t3C7Zx++ukppeTOO+9Mkuy+++55+ctfvsT677nnnowYMSInnXTSItvcf//9M3HixIwaNSove9nL8qMf/WiRxy3v99wBBmQGLSt/BnMGyZ9FyR/5061PGTRQjoFkUJfBmkHDhpWe3+NnTv1GXvHq7bP2hHWy+nPWyAu2eEmOP/FLS/y+L73utuz+prdkjTXHZuy4cXnjW/bNtbfdnST5v/sf61lu61dul61fud0ij73s+v/L2w58V9Zd/7lZbdSoTNvk+fnEZ05aYhsXXHld3rDP27POuutltVGjMvl5U3PkkR9JIoNWspV+DEQX+TO48mf/Qw/LsGFF/qx8XoetIjJIBg2GDGrFGcyzk2zY6/7kJH9twXp5Fm9605vyrne9K8ccc0wuuuiiHHfccRk2bFhmzpyZv/71r9luu+2y5ppr5pRTTsm4ceNy6qmnZo899sh5552X3XbbLUnyhje8IePHj8+XvvSlrLPOOrn77rtzwQUXLBJQvf3oRz/K7rvvnpe+9KWZOXNmkizzHZcDDzwwxx57bObMmZO11lqrZ/qZZ56ZtddeO7vvvnuS5He/+11e85rXZMstt8zXvva1rL766vnyl7+cnXbaKVdeeeVS/8B6u/322/OWt7wlRx99dIYNG5bLL7887373uzN37twcdthhPcu9+c1vznXXXZd///d/z6abbpqzzz47H/zgB5e7nx999NG8+tWvzty5czNz5sxMmzYtF110UQ4//PA89dRTPeu444478opXvCKrr756Zs2alU033TR33XVXfvaznyXp+sjFAQcckAULFuQrX/lKkq53BpfmoIMOyn777ZebbrppkXfSvvvd7yZJ9ttvvyTJXXfdlW222SbrrrtuTjzxxEycODFnn3129tlnn5x77rnZc889k/T/9zwIyaBVTP50kT/yp5sMWsVkUJe7/3Jndt59z7zr/R/JsGHD8tv/uTKzPvahPPnk3LztwH98DPOf3nNgbr35D/ngUf+WKVM3zkU//VE+/YmPLXc/P/b3R/POvXfNU0/OzeH/dFQmbfi8XHnZL/L/ffyjefrpp/OOQ96bJJn9lz/ngDe+LqPHjMnh/3R0pkzbOPf+9e7cfM2vksiglUz+rGLyp8tAz5+rLv9lEvmzCsigVUwGdZFBAzODWtFg/kmSD5RSzkrXBd0fcc2dle8973lPjj766CTJ61//+jz66KP53Oc+l4985CM54YQTMmfOnFx11VXZZJNNknS9G7L55pvnX//1X7PbbrvlgQceyJ/+9Kf8+Mc/7hmASfKOd7xjmdvccsstM2rUqKyzzjrL/cjE/vvvn3/913/NOeeck/e9731Jknnz5uWss87K29/+9qy22mpJkn/5l3/JlClT8stf/rJn2i677JItttgixx133CLvcC3Nxz/+8Z6fn3nmmcyYMSP33HNPvvSlL/U0eC6++OL86le/yve+973su+++PdvYbbfdMnv27Gdd/+c///n8+c9/zg033JBNN900SbLTTjvl4YcfzqxZs3L44YdnxIgROfbYYzN37txcd911ee5zn9vz+He+851Juj5yMXbs2MyfP3+5+26vvfbK2LFjc8YZZ+RTn/pUz/Qzzjgjr3/967Peeusl6Xq3sNaayy67LBMmTOh5XnfddVc+8YlPZM8992z0ex6EZNAqJn+6yB/5000GrWIyqMu7P/jRnp+feeaZTH/ldnngvnvzX2ec1vPi6qrLL8nvr/5NPn3K17PbXl1nFL16xuvy/gPfknvvefYewHe+8ZXcc/dd+f7Fv87zpm2cJNn2NTPy90cfzVdO/EzeduC7MmLEiHzphE/lySefzDkXXZF11//HdQT/7cjDk8iglUz+rGLyp8tAz58939rViJE/K50MWsVkUBcZNDAzaLmXyCilfC/JVUk2K6XMLqUcWko5rJSy8PSsC5LcnuS2JF9L8v6VVi093va2ty1yf999981jjz2WG2+8MZdffnm23XbbnlBJkuHDh2e//fbLtddem0cffTQTJkzIRhttlKOPPjpf+9rX8qc//aml9W244YbZYYcdcsYZZ/RMu/DCC/PAAw/koIMOSpLMnTs3l112Wd761rdm2LBhmT9/fubPn59aa3baaadcfvnlSZJaa8+8+fPnZ8GCBT3r/NOf/pT99tsvkyZNysiRIzNy5Mh8/etfzy233NKzzFVXXZXhw4cv8VGNhc2eZ3PhhRdmm222ybRp0xapYZdddsmDDz6Ym266KUnys5/9LG94wxsWae40NWbMmOyzzz75zne+0/NtrDfccEOuu+66nn23sLbdd98948aNW6K26667bpX8nlcFGTTwyJ8u8qfz8yeRQQORDOry5zv+L0cdcWh2mr55Xj5tYl4+bWJ++L1v587bb+tZ5rrfXZ3hw4dnp93/8eIiSXbdc8mPry7uyst+nhe/7OWZtOHzFqnhVTu8Ng/PeSi3/+mPSbpewG3/utcv8sKqKRm0KPkz8MifLvKn8/MnkUEDkQzqIoMGZgYtt8Fca92v1rpBrXVkrXVyrfUbtdYv11q/3D2/1lqPqLVuXGt9ca31mpVfNgvfvVj8/t13352HHnpoqd9Euf7666fWmjlz5qSUkosvvjjTp0/PMccck+c///nZaKON8qUvfallNR500EH59a9/nTvuuCNJ1zsvm2yySc87Nw899FAWLFiQ4447rqc5s/B2yimnZM6cOXnmmWfyrW99a5F5G2/c9Q7SY489lp133jnXXXddPv3pT+eKK67I1VdfnXe961156qmneuq45557stZaa2XkyJHPug+X5r777svll1++RH1vfetbkyQPPvhgz7+tvLj9QQcdlLvuuqvnukdnnHFG1lxzzey1116L1Pbtb397idr+5V/+paemVfF7Xtlk0MAjf+TPUMmfRAYNRDIoeeLxx3LYO/bOrTf/IR8+5hP55g8uyHfP+2Xe9PYD8nSvDHrg3r9l7LjxS2TQhHWWf/3Lhx54IL/9nyt7XrgtvP3zYQcnSR6eMydJ8sich7LeBpNWeJ8tJIP+Qf4MPPJH/gyV/Elk0EAkg2TQQM6gVlwigza49957s9FGGy1yP0kmTZqUtddeO3/729+WeMzf/va3lFKy9tprJ0k22mijfPvb306tNdddd11OOeWUvP/978/UqVN7rs+zIvbZZ58cccQROfPMM/PhD384P/3pT3PMMcf0zB8/fnyGDRuWI444YpF3ZHobNmxY3vjGN+bqq6/umTZq1KgkXWcG/vnPf84VV1yR7bbbrmf+4hct32CDDTJnzpzMmzdvkXBZuM+ezYQJE7Luuuvm85///FLnb7bZZknSc02bVtlhhx0yZcqUnHnmmdlhhx3yve99L295y1syZsyYRWp7zWtek6OOOmqp61h4NuPK/j0z9Mgf+SN/aKehnkE1yXW/vTp/nX1XvvmDC7LVK17ZM3/B6Ytm0DrrrZ9HH3l4iQx68IH7l/scxq21dl62zjr52MxPL3X+1I27zpAav/aE3Pe31l1yUwYxkMkf+SN/aCcZJIMGcgZpMA9S55xzTs+1d5LkrLPOyhprrJEtttgiO+ywQ0466aTceeedmTp1apJkwYIFOfvss7PllltmzTXXXGRdpZS87GUvywknnJBvfOMbufHGG5c54EaNGpW5c+f2qcaF77ScccYZee5zn5snn3wyBx54YM/85zznOXnNa16T6667LltttVWGDVv6CfUTJkzoubZMb0888USSLBIWc+bMyY9//ONFlnvlK1+ZBQsW5Ac/+MEiH0s/66yzlvscdt1113zhC1/IlClTsu666y5zude//vX54Q9/mHvuuWep7xomXfvu73//+3K3mXT9Tvbff/+ceuqp2XvvvTN79uwlwnfXXXfNVVddlRe96EWLBM6zrbOvv2d4NvJH/sgf2mmoZ9D1sx/Ok911jOiVQY8+/HAu/dkFiyz70q22zoIFC/LzC37Sc/3BJLnwJz9Y7nN49YzX5Xvf/GrWnzT5Wc/2eeX2O+YX/31e7r/3b5m43vpLXUYG0Snkj/yRP7STDJJBAzmDNJgHqa997Wt55plnsvXWW+eiiy7K17/+9cycOTPjx4/PkUcemdNPPz0777xzZs2albFjx+aLX/xibr311px//vlJkuuvvz4f/vCH8/a3vz2bbLJJFixYkNNPPz0jRozIa1/72mVud/PNN88VV1yR8847L+uvv37WWWednvBamoMOOijf+973cuyxx2a77bbLtGnTFpl/wgknZPvtt88uu+ySQw89NBtssEEeeOCB/O53v8uCBQvy6U8v/R2jJHnVq16VsWPH5ogjjsisWbPy+OOP5/jjj88666yTRx55pGe5nXfeOdttt13e97735YEHHsimm26as88+OzfeeONy9/ORRx6Zs88+O695zWty5JFHZrPNNsvjjz+eP/7xj7niiit6mkmzZs3K+eefn1e96lX5+Mc/nk022SR33313Lrzwwpx55pk9++6LX/xizj777Gy88cZZc801e85AXNa++9SnPpXDDjus51pGvX3yk5/MK17ximy//fb5wAc+kKlTp2bOnDm58cYbc/vtt+e0005r/HuGZyN/5I/8oZ1kUPLS6a/IGmuumU/967/k8I8enblPPJGvnfyfGb/WhPz90Ud7lnvl9jtmy623zfHHHJmH5zzY8w3qt91y83L38wHvPjwX/fRHOWSf3XPAuw/P1I03zdwnHs+dt/0pv/vfq/L507q+1fzwfzomV/ziZzlo713y7g/8UzaculHu+9tf8x//e4UMouPIn8GRP7++9Bc5/4fn9Ow7+UOnkEEyaEBnUK21LbeXv/zltT+ed9R5/b797Ne/rdfdNWeJ29+fnNevbbfa3KfnN37sscceW5PUG264oc6YMaOOHj26rrfeevXf/u3f6oIFC3qW++Mf/1j32muvOnbs2Dpq1Ki6zTbb1P/+7//umX/vvffWgw46qG666aZ1zJgxda211qrbb799vfDCC3uWueSSS2qSeskll/RMu/nmm+t2221Xx4wZU5PUd77znYvUtbj58+fX9ddfvyapX/nKV5b6nG666ab69re/vU6cOLGuttpqddKkSfWNb3xjPf/885e7P37xi1/Ul73sZXX06NF1o402qp///OeXWst9991X991337rGGmvUcePG1QMPPLCee+65Szy/HXbYoc6YMWORxz700EP1Ix/5SJ06dWodOXJknThxYt1uu+3qiSeeuMhyt912W913333rhAkT6mqrrVanTZtWP/KRj/TMv+eee+puu+1W11hjjZqk7rDDDrXWWr/5zW/WJPWOO+5Y4vlNnz69JqnHHHPMUp//XXfdVQ899ND63Oc+t44cObKuv/76daeddqpnnHFGrbVvv+dluemmm551fpJrapvyoxW3lZ1By8qfwZxB8mdR8kf+NL218xhIBg3+DFr4e/zqWT+um73oxXXUqNF18pSp9WOzPl0PO/KommSR3/cl1/6p7rrnm+vqz1mjrjl2bH3DPm+vJ339OzVJ/fo5P+1Zbvq2r67TX7ndIo+94oY76gGHHl6fu+GUOmLkyLrWhHXqlltvW//l2H9fZLnzrvhd3XXPN9fxa61dR662Wp005XkyaADf+ptBTSxrH67I66BWkD//0Mn5c8Chh/fUPNjyZ+Hv5tkMtQxyDCSDFpJBXQbqMVDpmr/qTZ8+vV5zTd+vAT/16PP7vY2v7blB1puy0fIXbLGXTB6/0tY9c+bMzJo1K/PmzcuIEU5Ab7WtttoqG220Ub7//e+3u5S2u/nmm/PCF75wmfNLKb+ttU5fhSW11MrOIPlDf8mff5A/i3IM1EUGdbl+9sMrZb1v322HTJ4yNZ/7yrdasr6VORZWNhm04pa3Dwcb+dNlsORPIoMGMq/D+k8GdZFBK9+K5M/QHZnQy+23357LLrss119/fd7xjne0uxxgCJE/QDvN/vOdueY3v86fbv5Ddt/rLe0uBxhC5A/QTjKotTSYIcnJJ5+cM844I/vvv3/e//73t7scYAiRP0A7ffebX8l5Pzg7u+/91rztnYe2uxxgCJE/QDvJoNZa+tc1MmDNnDkztdYh/bGIleGkk07Kgw8+mG9961tZffXV210ODEjyZ+WQP9A3Mmjl+NjMT+XyG27P8Sd+KWPGyCBYGvmzcsgf6BsZtHLIoNbSYAYAAAAAoBENZgAAAAAAGtFgBgAAAACgEQ1mAAAAAAAa0WAGAAAAAKARDWYAAAAAABoZ0e4C2uGFE1fLyFGrt6+AeU8mI0e3b/tAW8kgoJ1kENA27f77b/f2gbZyDAQrz5BsMI8ctXoyc1z7Cpj5SPu23cCdd96Z008/PQcddFA22mijlq770ksvzY477phLLrkkM2bMaOm6YaCSQX23KvNnwYIFOfHEE3PBBRfkD3/4Q5544olsuummOeKII3LIIYdk2DAf+qEzyKC+W9XHQCeeeGK++93v5vbbb8/jjz+eyZMnZ6+99srHP/7xTJgwoaXbh7YYOVr+9FE7X4M9/PDDecELXpB77703F198cXbaaaeWbh/axTFQ363qDJo5c2ZmzZq1xLJ77bVXzj333JZun5VjSDaY6Z8777wzs2bNynbbbdfyYAF4Nqsyf+bOnZvjjz8+Bx10UD784Q9njTXWyAUXXJD3vOc9+eMf/5jPfvazK3X7wMCzqo+BHnroobz5zW/OFltskTXXXDO///3v88lPfjKXXHJJrrnmGm90wRDSztdgRx11VEopq3SbwMDSrgz61a9+leHDh/fcX3vttVfZtlkxGsy0VK018+bNy2qrrdbuUoAhZkXzZ8yYMbn99tsXOYh53etelzlz5uQLX/hCPvnJT2bMmDGtKhfoMK04BjruuOMWuT9jxoysvvrqOeyww/L73/8+L3/5y1e0TKADtfI12K9//euceeaZ+cIXvpBDDz20BdUBna6VGbTNNttkxAitysHIaRCDzMyZM1NKyZ/+9KfsscceWWONNfK85z0vn/zkJ/PMM8/0LPfAAw/k8MMPz6RJkzJq1Ki84AUvyFe/+tWlrmtxBx98cKZOnZrkHx9dSJKdd945pZSUUnLppZcmSaZOnZoDDjggp512Wl7wghdktdVWy/nnn58kOfbYY7PVVltl3LhxWWeddfLa1742v/nNb1bCXgFWhU7Pn+HDhy/1HfKtt946Tz31VB544IE+7yug9To9g5Zl4aUxRo4c2ejxwIobKvkzb968vO9978vRRx/tk6swgAyVDGJw87bAILX33nvnkEMOyZFHHpmf/vSnOfbYY7PhhhvmkEMOyaOPPppXv/rVmTt3bmbOnJlp06bloosuyuGHH56nnnoqH/zgB/u8na222iqnnnpqjjjiiJx88snZeuutkySbb755zzKXXHJJrr322hx77LFZd911e0Lp7rvvzpFHHpnJkyfn8ccfz5lnnpntt98+11xzTV7ykpe0dH8Aq85Qy5/LLrss48ePzwYbbNCvxwErx1DIoPnz5+fpp5/O9ddfn2OPPTave93rHDvBANDp+fMf//Efefrpp/Oxj30sV111Vf93ELBSdXoGJcmGG26Y++67L5MnT86+++6bmTNn+hTpIKHBPEh99KMfzSGHHJIk2WmnnfLLX/4y3/ve93LIIYfk85//fP785z/nhhtuyKabbtqzzMMPP5xZs2bl8MMP7/NHDsaOHdsTIi984Quz7bbbLrHMnDlz8tvf/jbrr7/+ItO//vWv9/y8YMGC7LrrrnnRi16Ub3zjG/n85z/f6HkD7TeU8ueiiy7KOeeck+OOO85HtWCA6PQMeuyxx7Lmmmv23N9ll13yX//1X32qGVi5Ojl/brvtthx//PH5yU9+klGjRvWpTmDV6uQM2mSTTfLpT386W265ZUop+dnPfpYTTzwxv/vd73LxxRf3qW7ayyUyBqk99thjkftbbLFF/vKXvyRJLrzwwmyzzTaZNm1a5s+f33PbZZdd8uCDD+amm25qaS3bbrvtEqGSJD//+c+z4447ZsKECRkxYkRGjhyZW2+9NbfccktLtw+sWkMlf2666abst99+mTFjRo466qhWlg2sgE7PoNVXXz1XX311rrjiipx88sm59tpr88Y3vjHz589vae1A/3Vy/hx++OHZa6+9svPOO7e0TqB1OjmDDjjggBx11FF5/etfn5133jmf/exn89nPfjY///nP8/Of/7yltbNyOB1rkFr8OqGjRo3Kk08+mSS57777ctttty3zWn0PPvhgS2tZ2sfGf/e732X33XfPLrvskm984xvZYIMNMnz48Lz73e/uqRMYnIZC/tx+++3ZeeedM23atJx77rnOXoYBpNMzaNiwYZk+fXqSZLvttsuLX/zi7Ljjjvn+97+ffffdt6X1A/3Tqflzzjnn5Ne//nWuueaaPPzww0m6Pk2RJI8//ngeeeSRjBs3rqX1A/3XqRm0LPvtt18+8pGP5Oqrr85OO+3UirJZibxi7kATJkzIuuuuu8yPH2y22WZJktGjRydJnn766UW+7bO/wbO0C8T/4Ac/yIgRI/LDH/5wkYCbM2dOxo8f36/1A4NHJ+TP7Nmz87rXvS5jx47NhRdemLFjx/arJqB9OiGDFrew2Xzbbbf1+7HAqjOY8+emm27K3Llz86IXvWiJeW9605sybty4nsYzMDAN5gxqsi0GHg3mDrTrrrvmC1/4QqZMmZJ11113mcs973nPS5LceOON2WqrrZIkDz/8cK688spFrv238Bpcc+fO7XMNTzzxRIYPH75IEPzyl7/MX/7yl0ybNq1fzwcYPAZ7/tx///09745ffPHFmThxYp+3C7TfYM+gpbnsssuSJBtvvHG/HwusOoM5fw4++ODMmDFjkWnXXnttjjzyyPznf/5nttlmmz7XALTHYM6gZfnOd76TJDJokNBg7kBHHnlkzj777LzmNa/JkUcemc022yyPP/54/vjHP+aKK67Ij3/84yTJbrvtlnHjxuU973lPZs2alaeeeir/8R//kTXWWGOR9T3/+c/PiBEjctppp2XttdfOqFGjstlmmy0SPovbddddc9JJJ+Xggw/OIYcckltvvTXHHXdcJk2atFKfO9Begzl/5s6dm1122SV33nlnTjvttMyePTuzZ8/umb/55ps7mxkGuMGcQY888kh23XXX7L///tl0001TSsn//u//5oQTTshLX/rSvPnNb17xHQSsNIM5f6ZOnZqpU6cudd5LX/rSbLfddv3bGcAqN5gzKEm23HLLHHTQQdlss81SSsnFF1+cL3zhC9l1112z4447rtjOYZUYkg3meU89kZEzH2ljAU8mI0evtNWPGzcuV155ZT75yU/mM5/5TO6+++6MHz8+m222WfbZZ5+e5caPH5/zzjsvRx55ZN72trdl8uTJ+cQnPpGf//znufTSS3uWmzBhQk455ZR85jOfyQ477JAFCxbkkksuWeJd7t522WWXnHzyyTnhhBPygx/8IFtssUW+/e1v5/jjj19pzxsGi07OoMGcP/fee29+//vfJ0n233//JeYvb7swWMiggZlBo0ePzgtf+MKcfPLJufvuuzNixIhMnTo1H/3oR/OhD32o50wiGNTmPZnInwGXPzBUOAYauBm02Wab5ZRTTsk999yTBQsWZOONN84nPvGJfOxjH1uR3cIqVGqtbdnw9OnT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\n",
      "text/plain": [
       "<Figure size 1800x504 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_tcav_scores(experimental_sets, positive_interpretations, ['convs.1', 'convs.2'], score_type='sign_count')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now let's perform the same experiment for examples that have negative sentiment."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "neg_input_texts = [\"It was not a good movie pad\", \"I've never watched something as bad\", \\\n",
    "    \"It is a terrible movie ! pad\"]\n",
    "neg_input_text_indices = covert_text_to_tensor(neg_input_texts)\n",
    "\n",
    "\n",
    "negative_interpretations = tcav.interpret(neg_input_text_indices, experimental_sets=experimental_sets)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Visualizing TCAV scores for sentances that have negative sentiment. As we can see from the digram below, TCAV score for `Positive Adjectives` is relatively low comapare to the opposite `Neutral` concept. This observation holds cosistently true accross all experimental sets and layers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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P0gpJaWUHH744QvVNd9rX/vavPOd71yk/v/5n/9JKSU//elPFyy7/fbbs99++2X11VfPKquskje/+c258cYbe/R4XHzxxdltt90ybty4rLbaatlyyy3zve99b5HtZs6cmfe+970ZNWpUxowZk8MOOyxPPfXUItst7q0Rjz/+eI4++uiMHz8+w4cPz2abbZZzzz13ke/9y1/+kve///1Ze+21M3z48Gy00UY57hOfWLDf66+/Pr/5zW8WPHbzj9P9rRH77LNPtt5660X2/8gjj2To0KE5/fTTFzrmoYcemnHjxmX48OF54xvfmJ/85CcLfd+yfs8L6TKGoDv5s7CWz5/jjktefln+0DZk0MJaPoPMgfqD87B+In+Sl1+uC75u+fw57ri8/HIdOPkDyyCDFjYQMmgwnof15BIZy7K4C67UxSxLrfXcJOcmyaRJkxa7DT1zwAEH5IMf/GBOOumkXHvttTn11FOz0korZcqUKXn44Yez4447ZuTIkTnzzDMzevTonHXWWdl3331z5ZVXZu+9906SvO1tb8uYMWPyzW9+M2uuuWYeeuihXH311QsFVFc/+clPss8+++QNb3hDpkyZkiRLfMbl/e9/f04++eTMmjUrq6+++oLlF154YdZYY43ss88+SZJbb701O+20U7bccst8+9vfzqqrrppzzjknb33rW3PTTTct9g+sq+nTp+eggw7KiSeemJVWWik33HBDjjzyyMyePTtHHXXUgu3e8Y535Pbbb8+//uu/ZpNNNskll1ySY489dpmP8zPPPJM3v/nNmT17dqZMmZKJEyfm2muvzdFHH50XX3xxwT7+8pe/ZNttt82qq66aU045JZtsskkefPDB/PznP08evi1nTzk27zv20cybNy/f+tI/JUlGjVwtefi2ZNb9HQd79K5k5Vk57O1vySEfOyl3X/fjbP7qjRbU8h/nXJAkOWTXLZKHb8uDD/012+19aNZac4187bPHZdzY1XPJFT/PO9/5zvz0vNOy3547J0neNvmAjBk9Mt/8//4xa64xJg/99bFc/atf5+UZtyYrD1v4B153y2U+JrSJh2/r/ff8reMFUQe8fZ988D375aSPvCvXXndzR/4892imfOqoPPzXmdlxj/dk5GqvyJmn/kNGj1wtZ33v0o78+d7Xs/dub07SZVzKnyXqk/xZaSX5Q2uSQYMjg8yBVjTnYf1sMJ+DrbRSyR0znkqS3HTbXdlh933y7g8dk5VWWim/+5+bcuSRR+bPDz+Rd7//728DP/wd++XeP96VY0/452ww4VW59mc/yVEf+3iS5P9mPps1Ovf33IsdDY/5+3/2b8/kvW/bPS++MDsfOe4fM379DXPT9f+Vo48+OtMffSrvPeIjSZIZD9yf971994xYZZV8+JMnZIOJr8qjDz+Um2/4VVZaqeTss8/O+973vo78+da3knS8OnNxDjvssBxyyCG5++67F3o143/8x38kSQ455JAkyYMPPpjtttsua621Vr72ta9l3LhxueSSSzry56c/zX777Zek979nBhFzoMExBxqE52F90WCekWT9LvfXS/JwH+yXpfjwhz+cE088MUmy55575plnnslXv/rVfPKTn8xpp52WWbNm5eabb87GG2+cpOPZkM033zz/9E//lL333juPP/54/vznP+fyyy9f8J9gkrz3ve9d4jG33HLLDB8+PGuuueYy3zJx6KGH5p/+6Z9y6aWX5qMf/WiSZM6cObn44ovznve8JyuvvHKS5NOf/nQ22GCD/OpXv1qwbK+99soWW2yRU089daFnuBbnM5/5zIKvX+58huiRRx7JN7/5zQXB8otf/CK//vWvc9FFF+Xggw9ecIy99947M2bMWOr+v/71r+f+++/PH/7wh2yyySZJkre+9a156qmncsopp+Too4/O0KFDc/LJJ2f27Nm5/fbbs+666y74/g984APJw7dl81dvlFEjX5G5c+dl+61fv9Rj7r/nzhk1crVc8OOr8oWT/h5+F/z4quy58/Z55bixSZIpp30rtdZc/6NvZ+waYzp+rl3elAcffjSf/fI3s9+eO+fxJ2flz395IJf/+9cWBE2SvPfAvZdaAyzNh997YE485ogkyZ4775Bnnn0uX/3WhfnkkYfmtHMvzKyn/5abrzg/G0/cIEmyz+47ZvNdDso/fems7L3bmxcel12e4ZY/C+uT/EnkD21HBnUYEBlkDrSiOQ/rZ87BOhx57KcWfP3yyy9n0g475vHHHs0PLzhvQYP55hv+O7fd8tt88czvZO/9O7L2zbvsno+9/6A8+sjSh+kPvvutPPLQg/nRL36TDSe+Kkmy/U675G/PPJNvfe1Leff7P5ihQ4fmm6d9IS+88EIuvfbGrLX2369jut+7OprBm2++eUaNGpW5c+cu87Hbf//9M2rUqFxwwQX5whe+sGD5BRdckD333DOvfOUrk3S8YrPWmuuvvz5jx45d8Ng9+OCD+exnP5v99tuvod8z9IQ5UIcBMQfK4DsP64tLZFyR5LDOTzHePsnTrvu14r373e9e6P7BBx+cZ599NnfeeWduuOGGbL/99gsmNkkyZMiQHHLIIfn973+fZ555JmPHjs1GG22UE088Md/+9rfz5z//uU/rW3/99bPzzjvnggsuWLDsmmuuyeOPP57DDjssSTJ79uxcf/31ede73pWVVlopc+fOzdy5c1NrzVvf+tbccMMNSZJa64J1c+fOzbx5f//k1T//+c855JBDMn78+AwbNizDhg3Ld77zndxzzz0Ltrn55pszZMiQRd6qMT9kluaaa67Jdtttl4kTJy5Uw1577ZUnnngid999d5Lk5z//ed72trctFCqNWmWVEXnnPrvlB5f9ZzouqZf84Y9/zu1335vDDnrb32u77qbss9uOGT1qtYVr22WH3H73vXnmb89m7OpjstGG6+XEfz0j3/7BZfnz9AeWuz5499v3WOj+wfvtmWefez533jMtN/zPrdl+q9ctmNQknflzwF75/V33LDou5c8SyR9YPBnUQQbJoDgP63fOwTrc/5f/ywkf/1DeOmnzbD1xXLaeOC6XXfT93Dd92oJtbr/1lgwZMiRv3efvDdYkmbzfom+f7+6m63+Z171x64xff8OFanjTzrvlqVlPZvqf/5Sko4n9lt33XKi53KhVVlkl73znO/ODH/zg7/nzhz/k9ttvX/DYJR2P5z777JPRo0cvko233357v/yeGbzMgTqYA7XmHGiZDeZSykVJbk6yaSllRinlQ6WUo0op8193fnWS6UmmJfl2ko+tsGpZYP4zqN3vP/TQQ3nyyScX+0mUa6+9dmqtmTVrVkop+cUvfpFJkyblpJNOyqtf/epstNFG+eY3v9lnNR522GH5zW9+k7/85S9JOp793XjjjRc86/Xkk09m3rx5OfXUUxeEwvzbmWeemVmzZuXll1/O9773vYXWvepVHc9iP/vss9ljjz1y++2354tf/GJuvPHG3HLLLfngBz+YF198cUEdjzzySFZfffUMG7bw2wC6P4aL89hjj+WGG25YpL53vetdSZInnnhiwb99eXH7ww56Wx58+K+57qapSTqetRq52iuyf5dnnx57fFa+/6MrM2zDbRe6ffrU0ztqmvV0x+/5orMz6fWb56QvnJlX73RANtrh7fnm937YZ7Uy+Mx/9rT7/YceeSxPPvV01llrzUW+Z+1xa3bkz9PPLDwu5c8SyR9YPBkkgwZLBjkPaz3OwZLnn3s2R733wNz7x7ty3Emfzb//+Or8x5W/ygHveV9e6pI/jz/614waPWaR/Bm75rI/1OrJxx/P7/7npgXN6/m3fzjq8CTJU7NmJUmenvVkXrnOYi873pDDDjssDz744IJrz15wwQUZOXJk9t9//wXbPPbYY/n+97+/yGP36U9/OklHJvbH75nByRzIHKiV50DLvERGrfWQZayvST7eZxXRI48++mg22mijhe4nyfjx47PGGmvkr3/96yLf89e//jWllKyxxhpJko022ijf//73U2vN7bffnjPPPDMf+9jHMmHChAXXCFse73znO/Pxj388F154YY477rj87Gc/y0knnbRg/ZgxY7LSSivl4x//+ELPCne10kor5e1vf3tuueWWBcuGDx+epOMZqfvvvz833nhjdtxxxwXru1+0fJ111smsWbMyZ86chcJl/mO2NGPHjs1aa62Vr3/964tdv+mmmybJgmsX9ZWdd9g6G4xfOxdednV23mHrXPTTa3PQvrtnlVVG/L221Udnp223zAkfP3yx+1j3lR2Tt402XC/fP+PUjt/zXffmzPMvycc+84VMWH/dBddhgt54dOYT2WjD9Ra6nyTj11kra4wZnb/OfHyR7/nrzMc78mfM6CRdxuU6b5Q/SyB/YPFkkAwaLBnkPKz1DPZzsJrk9t/dkodnPJh///HV2WrbHRasn3f+wvmz5ivXzjNPP7VI/jzx+LI/ZHL06mvkjWuumX+c8sXFrp/wqo5XiY9ZY2we+2vfXRVm5513zgYbbJALL7wwO++8cy666KIcdNBBWWWVVRZsM3bs2Oy000454YQTFruP+a9kXNG/ZwYncyBzoFaeA/XFNZhpgksvvXTB9b+Sjk/RXG211bLFFltk5513zumnn5777rsvEyZMSJLMmzcvl1xySbbccsuMHDlyoX2VUvLGN74xp512Wr773e/mzjvvXGKwDB8+PLNnz+5RjfOf7b3ggguy7rrr5oUXXsj73//+Betf8YpXZKeddsrtt9+erbbaKiuttPgX1I8dO3bB9a26ev7555NkobCYNWtWLr/88oW222GHHTJv3rz8+Mc/XujtEBdffPEyf4bJkyfnG9/4RjbYYIOstdZaS9xuzz33zGWXXZZHHnlksa9cSJLhKw/L3559bpnHTDp+J4ceuHfO+t4Pc+DkXTPjkUcXeltEkkze5U25+Xd35LWv3mihwFnaPt+4xaY57eT/l+9e9NPcec+0AXFyReu59Ge/WHDtryS5+IqfZ7VXrJotNt04O2+/VU7/zkW578GHM2H9jgn2vHnzcskVP8+WW2yakau9YqF9yZ8lkz+weDJIBskgmmWwn4PdMeOpvNBZx9Au+fPMU0/lup9fvdC2b9hqm8ybNy+/vPqKBddgTpJrrvjxMn+GN++yey7693Oz9vj1lvqK5x3esmv+6z+vzMxH/5pxr1x7sdsMHz48f/vb35Z5zKQzfw49NGeddVYOPPDAzJgxY5EG2OTJk3PzzTfnta997UKN56Xts6e/Z1gWcyBzoFaeA2kwD1Df/va38/LLL2ebbbbJtddem+985zuZMmVKxowZk+OPPz7nn39+9thjj5xyyikZNWpUzj777Nx777256qqrkiR33HFHjjvuuLznPe/JxhtvnHnz5uX888/P0KFDs9tuuy3xuJtvvnluvPHGXHnllVl77bWz5pprLphALc5hhx2Wiy66KCeffHJ23HHHTJw4caH1p512Wt7ylrdkr732yoc+9KGss846efzxx3Prrbdm3rx5+eIXF/+sdZK86U1vyqhRo/Lxj388p5xySp577rl8/vOfz5prrpmnn356wXZ77LFHdtxxx3z0ox/N448/vuDTQ++8885lPs7HH398Lrnkkuy00045/vjjs+mmm+a5557Ln/70p9x4440LQuyUU07JVVddlTe96U35zGc+k4033jgPPfRQrrnmmlz4bx0fgrH5Jhvl7O//MJdcfm1eNWH9jHzFqtl046U8du96W75w5r/nqBP/Neuvu3Z23mHhT1L93KePyrb7Hpa3vPPIHHP4ezJh/XUz6+lncuef/i/TH5iR806bkjvuvjfHffYrec9+e2bjCetn3svzcv6lP+v4Pb95m2X+/LA43/6Pn3Tkzxtfm2uvuznf+Y+fZMqnPpoxo0fm+A+/L+df+rPscfDROeUfjsqo1V6Rs7//w9w7/YFc9f2OZ4AXGpeTnpA/S9An+XPhhR2PnfyhjcigAZRB5kC0Gedg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\n",
      "text/plain": [
       "<Figure size 1800x504 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_tcav_scores(experimental_sets, negative_interpretations, ['convs.1', 'convs.2'], score_type='sign_count')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Statistical significance testing of concepts"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In order to convince ourselves that our concepts truly explain our predictions, we conduct statistical significance tests on TCAV scores by constructing a number of experimental sets. In this case we look into the `Positive Adjective` concept and a number of `Neutral` concepts. If `Positive Adjective` concept is truly important in predicting positive sentiment in the sentence, then we will see consistent high TCAV scores for `Positive Adjective` concept across all experimental sets as apposed to any other concept.\n",
    "\n",
    "Each experimental set contains a random concept consisting of a number of random subsamples. In our case this allows us to estimate the robustness of TCAV scores by the means of numerous random concepts.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In addition, it is interesting to look into the p-values of statistical significance tests for each concept. We say, that we reject null hypothesis, if the p-value for concept's TCAV scores is smaller than 0.05. This indicates that the concept is important for model prediction.\n",
    "\n",
    "We label concept populations as overlapping if p-value > 0.05 otherwise disjoint.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_pval(interpretations, layer, score_type, alpha=0.05, print_ret=False):\n",
    "    P1 = extract_scores(interpretations, layer, score_type, 0)\n",
    "    P2 =  extract_scores(interpretations, layer, score_type, 1)\n",
    "    _, pval = ttest_ind(P1, P2)\n",
    "\n",
    "    relation = \"Disjoint\" if pval < alpha else \"Overlap\"\n",
    "\n",
    "    if print_ret:\n",
    "        print('P1[mean, std]: ', format_float(np.mean(P1)), format_float(np.std(P1)))\n",
    "        print('P2[mean, std]: ', format_float(np.mean(P2)), format_float(np.std(P2)))\n",
    "        print(\"p-values:\", format_float(pval))\n",
    "        print(relation)\n",
    "        \n",
    "    return P1, P2, format_float(pval), relation\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can present the distribution of tcav scores using boxplots and the p-values indicating whether TCAV scores of those concepts are overlapping or disjoint."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "def show_boxplots(layer, scores, metric='sign_count'):\n",
    "    def format_label_text(experimental_sets):\n",
    "        concept_name_list = [exp.name if i == 0 else \\\n",
    "                             exp.name.split('_')[0] for i, exp in enumerate(experimental_sets[0])]\n",
    "        return concept_name_list\n",
    "\n",
    "    n = 4\n",
    "    fs = 18\n",
    "    fig, ax = plt.subplots(1, 1, figsize = (20, 7 * 1))\n",
    "\n",
    "    esl = experimental_sets[:]\n",
    "    P1, P2, pval, relation = get_pval(scores, layer, metric)\n",
    "\n",
    "    ax.set_ylim([0, 1])\n",
    "    ax.set_title(layer + \"-\" + metric + \" (pval=\" + str(pval) + \" - \" + relation + \")\", fontsize=fs)\n",
    "    ax.boxplot([P1, P2], showfliers=True)\n",
    "\n",
    "    ax.set_xticklabels(format_label_text(esl), fontsize=fs)\n",
    "\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The box plots below visualize the distribution of TCAV scores for a pair of concepts in two different layers, `convs.2` and `convs.1`. Each layer is visualized in a separate jupyter cell. Below diagrams show that `Positive Adjectives` concept has TCAV scores that are consistently high across all layers and experimental sets as apposed to `Neutral` concept. It also shows that `Positive Adjectives` and `Neutral` are disjoint populations.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x504 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_boxplots(\"convs.2\", positive_interpretations, metric='sign_count')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x504 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_boxplots(\"convs.1\", positive_interpretations, metric='sign_count')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}