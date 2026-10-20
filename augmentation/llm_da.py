import json
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger('llm_da')

PROMPT_SPR = (
    "Please rewrite the following sentence using different sentence "
    "structures and vocabulary while preserving its core meaning and "
    "sentiment polarity: {text}"
)
PROMPT_SFR = (
    "Please flip the sentiment polarity of the following sentence from "
    "{orig_sentiment} to {target_sentiment}, ensuring the sentence is "
    "fluent and natural: {text}"
)

SENTIMENT_WORD = {-1: 'negative', 1: 'positive'}


class CLIPRouter:
    """CLIP text-video compatibility gate.

    phi(T, V) is the mean cosine similarity between the CLIP text embedding
    and the embeddings of F uniformly sampled frames. phi >= theta routes the
    sample to SPR, otherwise to SFR.
    """

    def __init__(self, model_name='openai/clip-vit-base-patch32', theta=0.25,
                 num_frames=8, device='cpu'):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.theta = theta
        self.num_frames = num_frames
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @staticmethod
    def sample_frames(frames, num_frames=8):
        idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
        return [frames[i] for i in idx]

    def compatibility(self, text, frames):
        import torch
        frames = self.sample_frames(frames, self.num_frames)
        with torch.no_grad():
            t_inputs = self.processor(text=[text], return_tensors='pt',
                                      padding=True, truncation=True).to(self.device)
            t_feat = self.model.get_text_features(**t_inputs)
            t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
            v_inputs = self.processor(images=frames, return_tensors='pt').to(self.device)
            v_feat = self.model.get_image_features(**v_inputs)
            v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
            scores = (v_feat @ t_feat.T).squeeze(-1)
        return float(scores.mean().cpu())

    def route(self, phi):
        return 'SPR' if phi >= self.theta else 'SFR'


class LLMRewriter:
    """LLM rewriting via the OpenAI chat API."""

    def __init__(self, model='gpt-3.5-turbo-0125', seed=42, temperature=0.7,
                 top_p=1.0, max_tokens=128, api_key=None, base_url=None,
                 dry_run=False):
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.dry_run = dry_run
        if not dry_run:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _call(self, prompt):
        if self.dry_run:
            return f'[dry-run] {prompt}'
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            seed=self.seed,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def rewrite_spr(self, text):
        return self._call(PROMPT_SPR.format(text=text))

    def rewrite_sfr(self, text, orig_sentiment):
        # orig_sentiment in {-1, +1}; the target polarity is its flip
        return self._call(PROMPT_SFR.format(
            text=text,
            orig_sentiment=SENTIMENT_WORD[orig_sentiment],
            target_sentiment=SENTIMENT_WORD[-orig_sentiment]))


def tokenize_like_reference(texts, reference_text_bert, tokenizer):
    """Tokenize rewritten texts into the (N, 3, seq_len) BERT layout
    [input_ids, attention_mask, token_type_ids]."""
    seq_len = reference_text_bert.shape[-1]
    encoded = tokenizer(list(texts), max_length=seq_len, padding='max_length',
                        truncation=True, return_tensors='np')
    out = np.stack([encoded['input_ids'], encoded['attention_mask'],
                    encoded['token_type_ids']], axis=1)
    return out.astype(np.float32)


def build_augmented_features(feature_path, output_path, rewrites, tokenizer):
    """D_aug = D + {(T'_i, V_i, A_i, y_i)}: each original training sample
    contributes one rewritten variant that retains the original audio/vision
    features and label. Only the train split is augmented."""
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)

    train = data['train']
    n_train = len(train['id'])
    assert len(rewrites) == n_train, \
        f'expected one rewrite per training sample ({n_train}), got {len(rewrites)}'

    aug_texts = [r['text_aug'] for r in rewrites]
    text_bert_aug = tokenize_like_reference(aug_texts, train['text_bert'],
                                            tokenizer)

    for key, aug_value in [
        ('text_bert', text_bert_aug),
        ('audio', train['audio']),
        ('vision', train['vision']),
        ('regression_labels', train['regression_labels']),
        ('audio_lengths', train.get('audio_lengths')),
        ('vision_lengths', train.get('vision_lengths')),
    ]:
        if key in train and aug_value is not None:
            data['train'][key] = np.concatenate(
                [train[key], np.array(aug_value)], axis=0)

    data['train']['raw_text'] = list(train['raw_text']) + aug_texts
    data['train']['id'] = [str(sid) for sid in train['id']] + \
                          [f'{sid}_aug' for sid in train['id']]
    for m in 'TAV':  # SIMS-style unimodal labels
        key = f'regression_labels_{m}'
        if key in train:
            data['train'][key] = np.concatenate([train[key], train[key]], axis=0)

    data['augmentation'] = [{
        'id': r['id'], 'branch': r['branch'], 'phi': r['phi'],
        'text_orig': t0, 'text_aug': r['text_aug'],
    } for r, t0 in zip(rewrites, train['raw_text'])]

    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    n_spr = sum(r['branch'] == 'SPR' for r in rewrites)
    logger.info(f'train split: {n_train} -> {2 * n_train} samples '
                f'({n_spr} SPR / {n_train - n_spr} SFR). Saved to {output_path}')


def run_llm_da(feature_path, output_path, frames_root=None, phi_cache=None,
               router=None, rewriter=None, tokenizer_name='bert-base-uncased',
               strategy='routed'):
    """Offline augmentation over the training split.

    strategy: 'routed' (CLIP gate selects SPR/SFR), 'spr' or 'sfr'
    (single-strategy ablation variants; the CLIP score is still recorded).
    """
    from transformers import BertTokenizer

    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    train = data['train']

    cache = {}
    if phi_cache and Path(phi_cache).is_file():
        cache = json.loads(Path(phi_cache).read_text())

    router = router or CLIPRouter()
    rewriter = rewriter or LLMRewriter()
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    rewrites = []
    for i, sid in enumerate(train['id']):
        text = train['raw_text'][i]
        label = float(np.asarray(train['regression_labels'][i]).reshape(-1)[0])

        if str(sid) in cache:
            phi = float(cache[str(sid)])
        else:
            if frames_root is None:
                raise ValueError(
                    f'no CLIP score cached for sample {sid} and no '
                    'frames_root provided')
            phi = router.compatibility(text, _load_frames(frames_root, sid))
            cache[str(sid)] = phi
        branch = router.route(phi) if strategy == 'routed' else strategy.upper()

        if branch == 'SPR':
            text_aug = rewriter.rewrite_spr(text)
        else:
            orig_sentiment = 1 if label > 0 else -1
            text_aug = rewriter.rewrite_sfr(text, orig_sentiment)

        rewrites.append({'id': sid, 'branch': branch, 'phi': phi,
                         'text_aug': text_aug})
        if (i + 1) % 50 == 0:
            logger.info(f'progress: {i + 1}/{len(train["id"])}')

    if phi_cache:
        Path(phi_cache).write_text(json.dumps(cache, indent=2))

    build_augmented_features(feature_path, output_path, rewrites, tokenizer)
    return rewrites


def _load_frames(frames_root, sid):
    """Frames of one clip: <frames_root>/<sid>.npz ('frames' array) or a
    directory <frames_root>/<sid>/ of images."""
    from PIL import Image
    root = Path(frames_root)
    npz = root / f'{sid}.npz'
    if npz.is_file():
        arr = np.load(npz)['frames']
        return [Image.fromarray(arr[j]) for j in range(arr.shape[0])]
    folder = root / str(sid)
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not files:
        raise FileNotFoundError(f'no frames found for sample {sid} under {root}')
    return [Image.open(p).convert('RGB') for p in files]
