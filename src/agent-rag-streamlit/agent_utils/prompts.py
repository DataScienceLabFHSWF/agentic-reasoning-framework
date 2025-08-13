"""
prompts.py
Promptvorlagen für den RAG-Chat-Workflow
"""

from langchain_core.prompts import ChatPromptTemplate


ROUTER_PROMPT = ChatPromptTemplate.from_template("""
Du analysierst, warum die abgerufenen Dokumente den Relevanzschwellenwert für eine RAG-Anfrage nicht erreicht haben.

Anfrage: {query}
Maximaler Relevanzwert: {max_score}
Schwellenwert: {threshold}

Die Dokumente wurden als nicht relevant genug eingestuft. Erstelle eine kurze Analyse, warum die abgerufenen Dokumente
möglicherweise nicht geeignet sind, um diese Anfrage zu beantworten. Dies dient dazu, die allgemeine Antwort zu verbessern.

Halte deine Analyse prägnant und sachlich.
""")


GENERAL_PROMPT = ChatPromptTemplate.from_template("""
Der Nutzer hat eine Frage gestellt, aber die abgerufenen Dokumente aus der Wissensdatenbank waren nicht relevant genug, 
um eine verlässliche Antwort zu geben.

Benutzeranfrage: {query}
Problem mit der Dokumentenrelevanz: {context}

Bitte antworte auf eine der folgenden Arten:
1. Falls du aus deinem allgemeinen Wissen antworten kannst, gib eine hilfreiche Antwort und erwähne, dass diese auf allgemeinem Wissen basiert, nicht auf den spezifischen Dokumenten.
2. Falls die Anfrage sehr spezifisch zu Dokumenten ist, die in der Wissensdatenbank vorhanden sein sollten, schlage dem Nutzer vor, die Frage umzuformulieren oder präzisere Begriffe zu verwenden.
3. Falls es sich um eine allgemeine Gesprächsanfrage handelt, antworte natürlich.

Sei hilfreich und ehrlich in Bezug auf die Einschränkungen.

Antwort:
""")

SUMMARIZER_PROMPT = ChatPromptTemplate.from_template("""
Basierend auf den folgenden, als hoch relevant eingestuften Dokumenten (Relevanzwert über dem Schwellenwert) 
erstelle bitte eine prägnante und genaue Antwort auf die Frage des Nutzers – zuerst auf Englisch, dann auf Deutsch.

Frage des Nutzers: {query}

Relevante Dokumente:
{context}

Anweisungen:
- Gib zuerst eine klare, prägnante Antwort auf Englisch
- Gib danach die gleiche Antwort auf Deutsch
- Beide Antworten basieren auf den Informationen in den Dokumenten
- Die Dokumente wurden als relevant für die Anfrage bestätigt
- Sei sachlich und zitiere spezifische Informationen aus den Dokumenten, wenn möglich
- Halte die Antwort dennoch gesprächig und hilfreich

Beispiel:
---
Frage des Nutzers: What does the German Atomic Energy Act say about decommissioning nuclear facilities?
Relevante Dokumente: 
"The Atomic Energy Act (Atomgesetz) of Germany requires that nuclear facilities may only be decommissioned once all nuclear fuel has been removed and a decommissioning license (Abbaugenehmigung) has been granted by the competent authority."

Antwort (Englisch):
Under the German Atomic Energy Act, nuclear facilities can only be decommissioned after all nuclear fuel has been removed and a decommissioning license has been issued by the competent authority.

Antwort (Deutsch):
Nach dem deutschen Atomgesetz dürfen kerntechnische Anlagen nur stillgelegt werden, wenn alle Kernbrennstoffe entfernt wurden und die zuständige Behörde eine Abbaugenehmigung erteilt hat.
---

Jetzt beantworte bitte die aktuelle Frage im gleichen Format.

Antwort (Englisch):

Antwort (Deutsch):
""")
