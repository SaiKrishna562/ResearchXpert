"""
rag.py — RAG engine for ResearchXpert (no eval metrics)
"""

from __future__ import annotations

import os, re, json, tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import requests


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PDFMeta:
    name: str
    pages: int
    chunks: int
    sections_found: list[str]


# ═══════════════════════════════════════════════════════════════════════════════
# Jina AI Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

class JinaEmbeddings(Embeddings):
    ENDPOINT  = "https://api.jina.ai/v1/embeddings"
    MODEL     = "jina-embeddings-v3"
    BATCH     = 64
    MAX_CHARS = 8000

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _embed_batch(self, texts: list[str], task: str) -> list[list[float]]:
        cleaned = [t[:self.MAX_CHARS].strip() or " " for t in texts]
        resp = requests.post(
            self.ENDPOINT,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.MODEL, "task": task, "input": cleaned},
            timeout=60,
        )
        resp.raise_for_status()
        items = sorted(resp.json()["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for i in range(0, len(texts), self.BATCH):
            results.extend(self._embed_batch(texts[i:i+self.BATCH], "retrieval.passage"))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text], "retrieval.query")[0]


def get_embeddings(jina_api_key: str) -> JinaEmbeddings:
    return JinaEmbeddings(api_key=jina_api_key)


# ═══════════════════════════════════════════════════════════════════════════════
# TOC-aware section parser
# ═══════════════════════════════════════════════════════════════════════════════

STANDARD_SECTIONS = [
    "abstract", "introduction", "related work", "background",
    "literature review", "methodology", "methods", "approach",
    "proposed method", "proposed model", "system design",
    "architecture", "model", "framework", "experimental setup",
    "experiments", "evaluation", "results", "findings",
    "discussion", "analysis", "conclusion", "conclusions",
    "future work", "limitations", "references", "bibliography",
    "acknowledgements", "acknowledgments", "appendix",
]


def extract_toc_sections(full_text: str) -> list[str]:
    toc_pattern = re.compile(
        r"(?:table\s+of\s+contents?|contents?)\s*\n(.*?)(?:\n\s*\n\s*\n|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    toc_match = toc_pattern.search(full_text[:8000])
    if not toc_match:
        return []

    toc_text = toc_match.group(1)
    sections = []
    line_re = re.compile(
        r"^\s*(?:chapter\s+\d+[:\.]?\s*|[\d]+(?:\.[\d]+)*\.?\s+)"
        r"([A-Za-z][^\.\n]{3,60}?)"
        r"(?:\s*\.{2,}\s*\d+\s*)?$",
        re.MULTILINE | re.IGNORECASE,
    )
    for m in line_re.finditer(toc_text):
        title = m.group(1).strip().lower()
        if len(title) > 3:
            sections.append(title)
    return sections


def build_section_regex(toc_sections: list[str]) -> re.Pattern:
    patterns = []
    for title in toc_sections:
        escaped = re.escape(title)
        patterns.append(r"(?:(?:chapter\s+\d+[:\.]?\s*|[\d]+(?:\.[\d]+)*\.?\s+))?" + escaped)
    patterns.append(r"(?:chapter\s+\d+[:\.]?\s+[A-Z][A-Za-z ,\-]{3,50})")
    patterns.append(r"(?:\d+(?:\.\d+)+\.?\s+[A-Z][A-Za-z ,\-]{3,50})")
    patterns.append(r"(?:\d+\.\s+[A-Z][A-Za-z ,\-]{3,50})")
    for name in STANDARD_SECTIONS:
        patterns.append(re.escape(name))
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(
        r"(?:^|\n)\s*(" + combined + r")\s*(?:\n|:)",
        re.IGNORECASE | re.MULTILINE,
    )


def split_into_sections(full_text: str) -> dict[str, str]:
    toc_sections = extract_toc_sections(full_text)
    section_re   = build_section_regex(toc_sections)
    matches = list(section_re.finditer(full_text))
    if not matches:
        return {"document": full_text}

    sections: dict[str, str] = {}
    preamble = full_text[:matches[0].start()].strip()
    if preamble:
        sections["preamble"] = preamble

    for i, m in enumerate(matches):
        title     = m.group(1).strip().lower()
        canonical = re.sub(r"^(?:chapter\s+\d+[:\.]?\s*|[\d]+(?:\.[\d]+)*\.?\s*)", "", title).strip()
        if not canonical:
            canonical = title
        start = m.end()
        end   = matches[i+1].start() if i+1 < len(matches) else len(full_text)
        text  = full_text[start:end].strip()
        if text:
            if canonical in sections:
                sections[canonical] += "\n\n" + text
            else:
                sections[canonical] = text

    return sections if sections else {"document": full_text}


def detect_section_from_question(question: str, available_sections: list[str]) -> Optional[str]:
    q = question.lower()
    for sec in available_sections:
        words = sec.split()
        if len(words) >= 2:
            for i in range(len(words) - 1):
                if words[i] + " " + words[i+1] in q:
                    return sec
        elif len(words) == 1 and len(words[0]) > 5 and words[0] in q:
            return sec

    keyword_map = [
        (["abstract"],                                           "abstract"),
        (["introduction"],                                       "introduction"),
        (["related work", "prior work", "literature"],          "related work"),
        (["methodology", "method", "approach", "proposed"],     "methodology"),
        (["architecture", "model specification", "design"],     "architecture"),
        (["overview of the proposed", "proposed model"],        "proposed model"),
        (["experiment", "experimental", "evaluation", "benchmark"], "experiments"),
        (["result", "findings", "performance", "accuracy"],     "results"),
        (["discussion", "analysis"],                            "discussion"),
        (["conclusion", "summary of"],                          "conclusion"),
        (["future work", "limitation"],                         "future work"),
        (["reference", "citation", "bibliography"],             "references"),
    ]
    for keywords, target in keyword_map:
        if any(kw in q for kw in keywords):
            for sec in available_sections:
                if target in sec or any(kw in sec for kw in keywords):
                    return sec
            return target
    return None


def structure_aware_split(
    pages: list[Document],
    filename: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> Tuple[list[Document], dict[str, str]]:

    page_offsets: list[Tuple[int, int]] = []
    full_text = ""
    for page in pages:
        page_offsets.append((len(full_text), page.metadata.get("page", 0)))
        full_text += page.page_content + "\n"

    def char_to_page(offset: int) -> int:
        pn = 0
        for off, pg in page_offsets:
            if offset >= off: pn = pg
            else: break
        return pn

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    sections = split_into_sections(full_text)
    chunks: list[Document] = []
    running_offset = 0

    for section_name, section_text in sections.items():
        label = section_name.title()
        for sub in splitter.split_text(section_text):
            idx = full_text.find(sub[:80], running_offset)
            approx_offset = idx if idx != -1 else running_offset
            chunks.append(Document(
                page_content=f"[{label}]\n{sub}",
                metadata={"source_file": filename, "section": section_name,
                          "page": char_to_page(approx_offset)},
            ))
        running_offset += len(section_text)

    return chunks, sections


# ═══════════════════════════════════════════════════════════════════════════════
# RAG Engine
# ═══════════════════════════════════════════════════════════════════════════════

class RAGEngine:
    MAX_PDFS = 3

    def __init__(self, groq_api_key: str, jina_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.groq_api_key  = groq_api_key
        self.jina_api_key  = jina_api_key
        self.model_name    = model_name
        self._vectorstore: Optional[FAISS] = None
        self._pdfs: List[PDFMeta]          = []
        self._sections: dict[str, dict[str, str]] = {}

    def _get_llm(self) -> ChatGroq:
        return ChatGroq(model=self.model_name, api_key=self.groq_api_key)

    @property
    def pdf_count(self)    -> int:           return len(self._pdfs)
    @property
    def pdfs(self)         -> List[PDFMeta]: return self._pdfs
    @property
    def total_chunks(self) -> int:           return sum(p.chunks for p in self._pdfs)
    @property
    def ready(self)        -> bool:          return self._vectorstore is not None

    def get_all_sections(self) -> list[str]:
        seen, result = set(), []
        for secs in self._sections.values():
            for k in secs:
                if k not in seen:
                    seen.add(k); result.append(k)
        return result

    def add_pdf(self, file_bytes: bytes, filename: str) -> PDFMeta:
        if self.pdf_count >= self.MAX_PDFS:
            raise ValueError(f"Maximum {self.MAX_PDFS} PDFs already loaded.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            pages  = PyPDFLoader(tmp_path).load()
            chunks, sections = structure_aware_split(pages, filename)
            self._sections[filename] = sections
            new_vs = FAISS.from_documents(chunks, get_embeddings(self.jina_api_key))
            if self._vectorstore is None:
                self._vectorstore = new_vs
            else:
                self._vectorstore.merge_from(new_vs)
            meta = PDFMeta(name=filename, pages=len(pages), chunks=len(chunks),
                           sections_found=list(sections.keys()))
            self._pdfs.append(meta)
            return meta
        finally:
            os.unlink(tmp_path)

    def reset(self) -> None:
        self._vectorstore = None
        self._pdfs        = []
        self._sections    = {}

    def remove_pdf(self, filename: str) -> None:
        self._pdfs = [p for p in self._pdfs if p.name != filename]
        self._sections.pop(filename, None)
        if not self._pdfs:
            self._vectorstore = None

    def answer(self, question: str, k: int = 6) -> tuple[str, list[Document]]:
        if not self.ready:
            raise RuntimeError("No PDFs loaded.")

        all_sections = self.get_all_sections()
        target = detect_section_from_question(question, all_sections)
        docs: list[Document] = []

        if target:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
            for filename, sections in self._sections.items():
                if target in sections:
                    for sub in splitter.split_text(sections[target]):
                        docs.append(Document(
                            page_content=f"[{target.title()}]\n{sub}",
                            metadata={"source_file": filename, "section": target, "page": "?"},
                        ))
                else:
                    for sec_name, sec_text in sections.items():
                        if target in sec_name or any(kw in sec_name for kw in target.split() if len(kw) > 4):
                            for sub in splitter.split_text(sec_text):
                                docs.append(Document(
                                    page_content=f"[{sec_name.title()}]\n{sub}",
                                    metadata={"source_file": filename, "section": sec_name, "page": "?"},
                                ))

        retriever = self._vectorstore.as_retriever(search_kwargs={"k": k if not docs else max(2, k // 2)})
        existing  = {d.page_content[:100] for d in docs}
        for d in retriever.invoke(question):
            if d.page_content[:100] not in existing:
                docs.append(d)

        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source_file','?')} | {d.metadata.get('section','').title()}]\n{d.page_content}"
            for d in docs
        )
        prompt = PromptTemplate.from_template("""\
You are ResearchXpert, an expert AI assistant for academic papers and theses.
The context contains labelled sections from the document.
Use the section content directly to answer the question.

Rules:
- Base your answer ONLY on the provided context.
- For section-specific questions, summarise from that section.
- If the answer is genuinely not present, say: "I couldn't find that in the document."
- Never hallucinate.

Context:
{context}

Question: {question}

Answer:""").format(context=context, question=question)

        return (self._get_llm() | StrOutputParser()).invoke(prompt), docs
