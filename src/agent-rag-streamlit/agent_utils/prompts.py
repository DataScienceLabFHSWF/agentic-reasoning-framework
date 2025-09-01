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


INTENT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_template("""
Du bist ein Assistent zur Intentionsklassifikation für ein spezialisiertes RAG-System über deutsche Kerntechnik und Nuklearsicherheit.

DOKUMENTENÜBERSICHT:
Unsere Wissensdatenbank enthält hauptsächlich:
- Sicherheitsberichte kerntechnischer Anlagen (z.B. KRB II - Kernkraftwerk Gundremmingen)
- Atomrechtliche Bestimmungen und Genehmigungsverfahren
- Technische Spezifikationen von Kernkraftwerken (Reaktortypen, Leistung, Blöcke)
- Stilllegungsverfahren und Rückbau kerntechnischer Anlagen
- Strahlenschutz und Überwachungsmaßnahmen
- Zwischen- und Endlagerung radioaktiver Abfälle
- Genehmigungsinhaber und Betreiber deutscher Kernkraftwerke
- Technische Sicherheitssysteme und Notfallmaßnahmen

AUFGABE:
Analysiere die folgende Benutzeranfrage und entscheide, ob sie für unsere spezialisierte Nuklear-Wissensdatenbank relevant ist.

Benutzeranfrage: {query}

BEWERTUNGSKRITERIEN:
- RELEVANT: Fragen zu deutschen Kernkraftwerken, Atomrecht, Nuklearsicherheit, Genehmigungen, Stilllegung, Atommüll
- NICHT RELEVANT: Allgemeine Gespräche, andere Energieformen, ausländische Nuklearanlagen, grundlegende Physik ohne Nuklearbezug

Antworte mit "RELEVANT" oder "NOT RELEVANT" und gib eine kurze Begründung.

Antwort:
""")


GENERAL_PROMPT = ChatPromptTemplate.from_template("""
Der Nutzer hat eine Frage gestellt, aber die Anfrage ist nicht relevant für unsere spezialisierte Nuklear-Wissensdatenbank, oder die abgerufenen Dokumente waren nicht relevant genug.

Benutzeranfrage: {query}
Kontext: {context}

ANWEISUNGEN:
1. Falls die Anfrage nicht nuklearspezifisch ist: Erkläre höflich, dass du auf deutsche Kernkraftwerke und Nuklearsicherheit spezialisiert bist
2. Falls die Dokumente nicht relevant genug waren: Schlage vor, die Frage spezifischer zu formulieren mit Begriffen wie:
   - Spezifische Anlagennamen (z.B. "KRB II", "Gundremmingen")
   - Atomrechtliche Begriffe (z.B. "Genehmigung", "Stilllegung", "Rückbau")
   - Technische Aspekte (z.B. "Reaktortyp", "Leistung", "Sicherheitssysteme")
3. Biete Beispiele für relevante Fragen an

Sei höflich, hilfreich und ermutige den Nutzer, spezifischere nuklearbezogene Fragen zu stellen.

Antwort:
""")

REASONING_PROMPT = ChatPromptTemplate.from_template("""
Du bist ein Experte für deutsche Kerntechnik und Nuklearsicherheit. Analysiere die folgenden Dokumente sorgfältig und beantworte die Frage des Nutzers basierend auf den bereitgestellten Informationen.

Nutze deine Denkfähigkeiten, um:
1. Die relevanten Informationen aus den Dokumenten zu identifizieren
2. Verbindungen zwischen verschiedenen Dokumenten herzustellen
3. Eine fundierte, sachliche Antwort zu formulieren

Frage des Nutzers: {query}

Verfügbare Dokumente:
{context}

ANWEISUNGEN:
- Antworte ausschließlich auf Deutsch
- Stütze deine Antwort auf die Informationen in den Dokumenten
- Analysiere die Dokumente gründlich und ziehe logische Schlussfolgerungen
- Sei präzise und verwende deutsche Fachbegriffe für nukleartechnische Konzepte
- Falls die Dokumente widersprüchliche Informationen enthalten, weise darauf hin
- Erkläre deine Denkweise, wenn dies zum Verständnis beiträgt

Antwort:
""")


SUMMARIZER_PROMPT = ChatPromptTemplate.from_template("""
Du bist ein Experte für deutsche Kerntechnik und Nuklearsicherheit. Erstelle basierend auf der detaillierten Reasoning-Antwort eine prägnante und benutzerfreundliche Antwort auf Deutsch.

Frage des Nutzers: {query}

Detaillierte Reasoning-Antwort:
{reasoning_answer}

ANWEISUNGEN:
- Antworte ausschließlich auf Deutsch
- Fasse die wichtigsten Punkte aus der Reasoning-Antwort zusammen
- Mache die Antwort für den Benutzer verständlich und gut lesbar
- Behalte wichtige technische Details und Zahlen bei
- Verwende deutsche Fachbegriffe für nukleartechnische Konzepte
- Strukturiere die Antwort klar und logisch

Zusammengefasste Antwort:
""")

FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
Basierend auf der folgenden zusammengefassten Antwort, gib eine finale, extrem prägnante Antwort auf die ursprüngliche Frage.
Deine Antwort sollte so kurz wie möglich sein, idealerweise nur ein oder zwei Wörter, eine Zahl oder eine Entität.
Füge keinen zusätzlichen Text oder Erklärungen hinzu. Der Zweck dieser Ausgabe ist die automatisierte Auswertung, nicht die menschliche Lesbarkeit.

Ursprüngliche Frage: {query}
Zusammengefasste Antwort: {summarized_answer}

Deine Antwort sollte so kurz wie möglich sein, idealerweise nur ein oder zwei Wörter, eine Zahl oder eine Entität.                                                       
Ursprüngliche Frage: {query}
Finale Antwort:
""")