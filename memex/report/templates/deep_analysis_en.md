# Deep Analysis Report Template

> Variables rendered by `report/agent.py`:
> `{query}`, `{date}`, `{executive_summary}`, `{news_findings}`,
> `{social_sentiment}`, `{expert_analysis}`, `{knowledge_context}`,
> `{forum_synthesis}`, `{key_entities}`, `{conclusion}`, `{sources}`

---

## {query}
*Generated {date} · Memex Research Engine*

---

### Executive Summary
{executive_summary}

---

### 1. Recent Developments
*Source: NewsAgent — NewsAPI + Jina Reader*

{news_findings}

---

### 2. Community & Practitioner Sentiment
*Source: SocialAgent — Reddit + Hacker News*

{social_sentiment}

---

### 3. Expert & Academic Analysis
*Source: ExpertAgent — Substack + arXiv + Jina Search*

{expert_analysis}

---

### 4. Foundational Context
*Source: KnowledgeAgent — Offline Corpus (pgvector RAG)*

{knowledge_context}

---

### 5. Adversarial Synthesis
*Multi-agent debate: Optimist · Pessimist · Realist · Host*

{forum_synthesis}

---

### Key Entities & Relationships
{key_entities}

---

### Conclusion
{conclusion}

---

### Sources
{sources}
