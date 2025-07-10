import re
import requests
import numpy as np
from typing import List
from collections import Counter, defaultdict

            
def has_repetitions(text, window_size=40, min_repeats=30):

    substr_counter = Counter()
    text = re.sub(r'\s+', ' ', text)
    substr_positions = defaultdict(list)

    if len(text) < window_size:
        print("Find a short text: ", text)
        return False

    for i in range(len(text) - window_size + 1):
        substr = text[i:i + window_size]
        substr_counter[substr] += 1
        substr_positions[substr].append(i)

    most_common = substr_counter.most_common(1)
    if not most_common:
        return False

    most_substr, count = most_common[0]

    return count >= min_repeats


def split_by_keywords(response, answer):
    match = re.search(rf'(?<![\d+\-*/^<>%&|!()[\],.?]){re.escape(answer)}\b', response.lower())
    end = match.end() if match else None
        
    if len(answer) > 3:
        answer_sci = answer[:-3] + "," + answer[-3:]
        match_sci = re.search(rf'(?<![\d+\-*/^<>%&|!()[\],.?]){re.escape(answer_sci)}\b', response.lower())
        if match_sci:
            end = match_sci.end() if end is None else min(match.end(), match_sci.end())
    
    if end is not None:
        return response[: end], response[end: ]
    
    start = response.lower().find(answer.lower())
    if start != -1:
        end = start + len(answer)
        return response[:end], response[end:]
    return None, None 


def split_cot(problem, cot, answer):
    fcs, aft = split_by_keywords(cot, answer)
    # if fcs is None or not answer.isdigit() or len(answer) <= 2:
    #     fcs, aft = split_by_llm(problem, cot, answer)

    return fcs, aft


def split_sentences(text: str) -> List[str]:

    latex_pattern = (
        r'(\$\$.*?\$\$|'            # $$...$$
        r'\$(?!\s)[^$]*?\$|'        # $...$
        r'\\\[.*?\\\]|'             # \[...\]
        r'\\\(.*?\\\)|'             # \(...\)
        r'\\begin\{[^{}]+\}.*?\\end\{[^{}]+\})'  # \begin{env}...\end{env}
    )
    latex_blocks = []

    def replace_latex(match):
        idx = len(latex_blocks)
        latex_blocks.append(match.group(0))
        return f"__LATEX_{idx}__"

    text_with_placeholders = re.sub(latex_pattern, replace_latex, text, flags=re.DOTALL)


    def insert_split_marker(text):
        result = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch in ".!?":

                if ch == '.' and i > 0 and i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
                    result.append(ch)
                else:
                    result.append(ch)
                    result.append("__SPLIT__")
            else:
                result.append(ch)
            i += 1
        return ''.join(result)

    marked_text = insert_split_marker(text_with_placeholders)


    raw_chunks = [chunk.strip() for chunk in marked_text.split("__SPLIT__") if chunk.strip()]


    results = []
    for chunk in raw_chunks:
        for idx, latex in enumerate(latex_blocks):
            chunk = chunk.replace(f"__LATEX_{idx}__", latex)
        results.append(chunk)

    return results


def cosine_similarity(a, b):

    a, b = np.array(a), np.array(b)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    # remove conditions similarity < 0
    return np.clip(np.dot(a, b), 0.0, 1.0)


def get_embedding_from_server(text: str, server_url):
    payload = {"text": text}
    try:
        response = requests.post(server_url, json=payload)
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def get_reasoning_gain(response_slices, server_url="http://localhost:8000/embedding", window_ratio=0.1, stride_ratio=0.05):
    if len(response_slices) == 1:
        return 0.0
    num_sentences = len(response_slices)
    window_size = max(1, int(window_ratio * num_sentences))
    stride = max(1, int(stride_ratio * num_sentences))

    def sliding_window_chunks(sentences, window_size, stride):
        chunks = []
        progress_x = []
        for i in range(0, len(sentences) - window_size + 1, stride):
            chunk_text = " ".join(sentences[i:i + window_size])
            progress = (i + window_size / 2) / len(sentences)
            chunks.append(chunk_text)
            progress_x.append(progress)
        return chunks, progress_x

    chunks, progress_x = sliding_window_chunks(response_slices, window_size, stride)

    chunk_embeddings = []
    for chunk_text in chunks:
        emb = get_embedding_from_server(chunk_text, server_url)
        chunk_embeddings.append(emb)


    redundant_spans = []
    adjacent_similarities = []
    for i in range(len(chunk_embeddings) - 1):
        sim = cosine_similarity(chunk_embeddings[i], chunk_embeddings[i + 1])
        if sim > 0.85:
            redundant_spans.append((i, i + 1))
        adjacent_similarities.append(sim)


    merged_segments = []
    if redundant_spans:
        current_start = redundant_spans[0][0]
        current_end = redundant_spans[0][1]

        for (i, j) in redundant_spans[1:]:
            if i == current_end:
                current_end = j
            else:
                merged_segments.append((current_start, current_end))
                current_start, current_end = i, j
        merged_segments.append((current_start, current_end))


    merged_results = []
    for seg_start, seg_end in merged_segments:
        covered_sentences = []
        seen = set()

        for win_idx in range(seg_start, seg_end + 1):
            start_idx = win_idx * stride
            end_idx = min(start_idx + window_size, len(response_slices))
            for s in response_slices[start_idx:end_idx]:
                if s not in seen:
                    covered_sentences.append(s)
                    seen.add(s)

        step_start_idx = seg_start * stride
        step_end_idx = min((seg_end + 1) * stride + window_size, len(response_slices))

        merged_results.append({
            "start_window": seg_start,
            "end_window": seg_end,
            "progress": float((step_start_idx + step_end_idx) / 2 / len(response_slices)),
            "redundant_text": covered_sentences
        })
    if not adjacent_similarities:
        return 0.0

    return 1 - np.array(adjacent_similarities).mean()
